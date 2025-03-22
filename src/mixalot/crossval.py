"""Code for cross-validation functionality."""
import json
from typing import List, Tuple
import os

import numpy as np
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold

from mixalot.datasets import MixedDataset
from mixalot.helpers import (
    combine_features,
    convert_categories_to_codes,
    parse_numeric_variable,
)
from mixalot.models import RandomForestSpec
from mixalot.trainers import train_random_forest

 

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


def load_cross_validation_folds_spec_from_json(
        json_path: str
    ) -> 'CrossValidationFoldsSpec':
    """Load a cross-validation folds specification from a JSON file.
    
    Args:
        json_path (str): Path to a JSON file containing the cross-validation
            folds specification.
            
    Returns:
        CrossValidationFoldsSpec: A cross-validation folds specification object
            created from the JSON file.
            
    Raises:
        FileNotFoundError: If the provided JSON file does not exist.
        ValueError: If the JSON file is missing required fields or contains
            invalid values.
    """
    # Ensure the file exists
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"Cross-validation specification file '{json_path}' does not "
            "exist."
        )
    
    # Load the JSON file
    with open(json_path, 'r') as file:
        spec_dict = json.load(file)
    
    # Check for required fields
    required_fields = ['n_splits', 'random_state']
    for field in required_fields:
        if field not in spec_dict:
            raise ValueError(
                "Cross-validation specification is missing required field: "
                f"'{field}'"
            )
    
    # Get values from the dictionary
    n_splits = spec_dict['n_splits']
    random_state = spec_dict['random_state']
    
    # Validate values
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError(
            f"n_splits must be an integer >= 2, got {n_splits}"
        )
    
    if not isinstance(random_state, int):
        raise ValueError(
            f"random_state must be an integer, got {random_state}"
        )
    
    # Create and return the specification
    return CrossValidationFoldsSpec(
        n_splits=n_splits,
        random_state=random_state
    )


def fit_single_fold(
    dataframe,
    dataset_spec,
    cv_spec,
    model_spec,
    fold_idx,
    random_seed=None
):
    """Fit a model on a single fold and calculate test loss.
    
    This function takes a dataframe, splits it according to the specified
    fold index from the cross-validation specification, processes the data
    into a MixedDataset, trains the model on the training set, and
    evaluates it on the test set.
    
    Args:
        dataframe (pd.DataFrame): The data containing all variables specified
            in dataset_spec.
        dataset_spec (DatasetSpec): The specification describing the variables
            in the dataset.
        cv_spec (CrossValidationFoldsSpec): The specification for
            cross-validation folds.
        model_spec (RandomForestSpec): The model specification.
        fold_idx (int): The index of the fold to use (0 to n_splits-1).
        random_seed (int, optional): Seed for reproducibility.
        
    Returns:
        dict: A dictionary containing:
            - 'model': The trained model
            - 'fold_loss': The loss on the test set
            - 'test_indices': The indices of the test set samples
            - 'predictions': Dictionary mapping test indices to predicted
                values (probabilities for classification, direct predictions for
                regression)
    """
    # Validate fold index
    if fold_idx < 0 or fold_idx >= cv_spec.n_splits:
        raise ValueError(
            f"fold_idx must be between 0 and {cv_spec.n_splits - 1}, "
            f"got {fold_idx}"
        )
    
    # Validate that the dataframe contains all required columns
    required_columns = list(dataset_spec.all_var_names)
    missing_columns = [col for col in required_columns 
                      if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            f"Dataframe is missing required columns: {missing_columns}"
        )
    
    # Get the number of samples
    num_obs = len(dataframe)
    
    # Get train and test indices for the specific fold
    train_indices, test_indices = cv_spec.get_fold_indices(fold_idx, num_obs)
    
    # Split the dataframe
    train_df = dataframe.iloc[train_indices]
    test_df = dataframe.iloc[test_indices]
    
    # Convert train and test dataframes to MixedDataset objects
    train_dataset = dataframe_to_mixed_dataset(train_df,
                                               dataset_spec, model_spec)
    test_dataset = dataframe_to_mixed_dataset(test_df, dataset_spec,
                                              model_spec)
    
    # Is this a classifier (categorical or ordinal) or regressor fit?
    y_var = model_spec.y_var
    y_var_spec = dataset_spec.get_var_spec(y_var)
    y_var_type = y_var_spec.var_type
    is_classifier = y_var_type in ['categorical', 'ordinal']

    # Get test features and target
    test_Xcat, test_Xord, test_Xnum, test_y = test_dataset.get_arrays()
    test_features = combine_features(test_Xcat, test_Xord, test_Xnum)
 
    # Convert target to numpy for consistent processing
    train_y_np = train_dataset.y_data.cpu().numpy()
    test_y_np = test_dataset.y_data.cpu().numpy()

    # Check if training data contains all categories present in test data
    if is_classifier:
        # Convert to numpy for consistency
        # TODO: we could check for this consistency for all folds prior to
        #       fitting any of the folds.
        test_categories = set(np.unique(test_y_np))
        train_categories = set(np.unique(train_y_np))
        
        # Find categories in test set that aren't in training set
        missing_categories = test_categories - train_categories
        
        if missing_categories:
            raise ValueError(
                f"Test set contains categories {missing_categories} that "
                f"are not present in the training set. Consider using a "
                f"different fold split or stratified cross-validation."
            )

    # Train the model
    if isinstance(model_spec, RandomForestSpec):
        model = train_random_forest(train_dataset, model_spec, random_seed)
    else:
        raise ValueError(
            f"Unsupported model type: {type(model_spec).__name__}"
        )
   
    # Ensure target is properly shaped
    if test_y.ndim > 1 and test_y.shape[1] == 1:
        test_y = test_y.squeeze()
    
    if is_classifier:
        # For classification models, get predicted probabilities
        y_pred_proba = model.predict_proba(test_features)
        
        # Get classes for log_loss calculation
        classes = model.classes_
        fold_loss = log_loss(test_y_np, y_pred_proba, labels=classes)
       
        # Store predictions (probabilities for classification)
        predictions = {}
        for i, idx in enumerate(test_indices):
            predictions[int(idx)] = y_pred_proba[i]
    else:
        # For regression models, get direct predictions
        y_pred = model.predict(test_features)
        
        # Calculate regression loss (mean squared error)
        fold_loss = mean_squared_error(test_y_np, y_pred)
        
        # Store predictions (direct values for regression)
        predictions = {}
        for i, idx in enumerate(test_indices):
            predictions[int(idx)] = y_pred[i]
    
    # Return model, loss, and predictions
    return {
        'model': model,
        'fold_loss': fold_loss,
        'test_indices': test_indices,
        'predictions': predictions
    }


def dataframe_to_mixed_dataset(
    df,
    dataset_spec,
    model_spec,
):
    """Convert a dataframe to a MixedDataset.
    
    Args:
        df (pd.DataFrame): DataFrame containing data for all variables 
            in dataset_spec.
        dataset_spec (DatasetSpec): DatasetSpec describing the variables 
            and their types.
        model_spec (RandomForestSpec): Model specification with y_var.
        
    Returns:
        MixedDataset: A dataset created from the dataframe.
    """
   
    # Initialize empty lists for each variable type
    cat_data = []
    ord_data = []
    num_data = []
    
    # Process each variable type
    for var_type, data_list in [
        ('categorical', cat_data),
        ('ordinal', ord_data),
        ('numerical', num_data)
    ]:
        ordered_vars = dataset_spec.get_ordered_variables(var_type)
        
        for var_name in ordered_vars:
            var_spec = dataset_spec.get_var_spec(var_name)
            
            # Get the column name to use (might be different from var_name)
            col_name =\
                var_spec.column_name if var_spec.column_name else var_name
            
            if col_name not in df.columns:
                raise ValueError(
                    f"Variable '{var_name}' (column '{col_name}') not found "
                    "in dataframe"
                )
            
            # Get the column data
            column_data = df[col_name]
            
            # Process the column according to its type
            if var_type in ['categorical', 'ordinal']:
                processed_column = convert_categories_to_codes(
                    column_data, var_spec)
            else:  # var_type == 'numerical'
                processed_column = parse_numeric_variable(column_data,
                                                          var_spec)
            
            # Append the processed column to the appropriate list
            data_list.append(processed_column)
    
    # Convert lists to numpy arrays
    Xcat = np.column_stack(cat_data) if cat_data else None
    Xord = np.column_stack(ord_data) if ord_data else None
    Xnum = np.column_stack(num_data) if num_data else None
    
    # Create and return the MixedDataset
    return MixedDataset(
        dataset_spec=dataset_spec,
        Xcat=Xcat,
        Xord=Xord,
        Xnum=Xnum,
        model_spec=model_spec
    )
