"""Code for cross-validation functionality."""
from __future__ import annotations
import json
import pickle
from typing import Dict, List, Optional,Tuple
import os

import pandas as pd
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
                values (probabilities for classification, direct predictions
                for regression)
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


def _fit_model(
    train_dataset: 'MixedDataset',
    model_spec: 'SingleTargetModelSpec',
    random_seed: Optional[int]
):
    """Fit a model based on the provided specification.
    
    Args:
        train_dataset: The training dataset.
        model_spec: The model specification.
        random_seed: Seed for reproducibility.
        
    Returns:
        Trained model object.
        
    Raises:
        ValueError: If model type is not supported.
    """
    if isinstance(model_spec, RandomForestSpec):
        return train_random_forest(train_dataset, model_spec, random_seed)
    else:
        raise ValueError(
            f"Unsupported model type: {type(model_spec).__name__}"
        )


def _calculate_losses(
    model,
    features,
    true_values,
    is_classifier: bool
):
    """Calculate predictions and losses for a set of observations.
    
    Args:
        model: The trained model.
        features: The feature matrix.
        true_values: The true target values.
        is_classifier: Whether this is a classification problem.
        
    Returns:
        Tuple containing (predictions, losses).
        For classification: predictions is a list of probability arrays.
        For regression: predictions is a list of scalar values.
        Losses is a list of loss values (one per observation).
    """
    predictions = []
    losses = []
    
    if is_classifier:
        # For classification models, get predicted probabilities
        pred_proba = model.predict_proba(features)
        classes = model.classes_
        
        # Process each observation
        for i in range(len(true_values)):
            true_label = true_values[i]
            probs = pred_proba[i]
            
            # Store prediction
            predictions.append(probs)
            
            # Calculate log loss for this observation
            true_label_array = np.array([true_label])
            pred_proba_array = np.array([probs])
            loss = log_loss(true_label_array, pred_proba_array, labels=classes)
            losses.append(loss)
    else:
        # For regression models, get direct predictions
        pred_values = model.predict(features)
        
        # Process each observation
        for i in range(len(true_values)):
            true_value = true_values[i]
            pred_value = pred_values[i]
            
            # Store prediction
            predictions.append(pred_value)
            
            # Calculate squared error for this observation
            loss = (true_value - pred_value) ** 2
            losses.append(loss)
    
    return predictions, losses


def _process_fold_observations(
    model,
    dataset,
    indices,
    fold_idx,
    sample_type,
    dataframe_indices,
    is_classifier: bool
):
    """Process observations for a fold and calculate predictions and losses.
    
    Args:
        model: The trained model.
        dataset: The dataset containing the observations.
        indices: The indices of the observations in the original dataframe.
        fold_idx: The fold index.
        sample_type: 'train' or 'test'.
        dataframe_indices: The original indices from the dataframe.
        is_classifier: Whether this is a classification problem.
        
    Returns:
        List of dictionaries, each containing results for one observation.
    """
    # Get features and target
    Xcat, Xord, Xnum, y = dataset.get_arrays()
    features = combine_features(Xcat, Xord, Xnum)
    y_np = dataset.y_data.cpu().numpy()
    
    # Ensure target is properly shaped
    if y.ndim > 1 and y.shape[1] == 1:
        y_np = y_np.squeeze()
    
    # Calculate predictions and losses
    predictions, losses = _calculate_losses(
        model, features, y_np, is_classifier
    )
    
    # Create result rows
    results = []
    for i, idx in enumerate(indices):
        original_idx = dataframe_indices[idx]
        true_value = y_np[i]
        
        # Create base result row
        result_row = {
            'original_index': original_idx,
            'fold': fold_idx,
            'sample_type': sample_type,
            'actual_value': true_value,
            'loss': losses[i]
        }
        
        # Add prediction details based on problem type
        if is_classifier:
            # For classification, add probabilities and predicted class
            probs = predictions[i]
            for j, prob in enumerate(probs):
                result_row[f'pred_prob_class_{j}'] = prob
            
            # Add predicted class (highest probability)
            classes = model.classes_
            result_row['pred_class'] = classes[np.argmax(probs)]
        else:
            # For regression, add the predicted value
            result_row['prediction'] = predictions[i]
        
        results.append(result_row)
    
    return results


def run_cross_validation(
    dataframe: 'DataFrame',
    dataset_spec: 'DatasetSpec',
    cv_spec: 'CrossValidationFoldsSpec',
    model_spec: 'SingleTargetModelSpec',
    random_seed: Optional[int] = None,
    output_folder: Optional[str] = None,
    overwrite_files: bool = False,
    rerun_folds: bool = False
) -> Dict:
    """Perform complete cross-validation using the specified parameters.
    
    This function runs cross-validation for each fold and records individual
    observation losses for both in-sample (train) and out-of-sample (test)
    predictions.
    
    Args:
        dataframe: The data containing all variables specified in dataset_spec.
        dataset_spec: The specification describing the variables in the dataset.
        cv_spec: The specification for cross-validation folds.
        model_spec: The model specification.
        random_seed: Seed for reproducibility.
        output_folder: Folder to save results to. If None, results aren't saved.
        overwrite_files: If True, overwrites existing files. Default is False.
        rerun_folds: If True, reruns all folds. Default is False.
            
    Returns:
        dict: A dictionary containing:
            - 'fold_results': List of results from each fold
            - 'observation_results': DataFrame with all observations and their
              predictions including sample_type ('train' or 'test'), actual
              values, predictions, and individual losses for each fold
            - 'test_avg_loss': Average loss across all out-of-sample observations
            - 'train_avg_loss': Average loss across all in-sample observations
    """
    # Validate input parameters
    _validate_cv_inputs(dataframe, dataset_spec, cv_spec, model_spec)
    
    # Create output folder if it doesn't exist
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Initialize containers for results
    fold_results = []
    observation_results = []
    
    # Get y variable type to determine loss type
    y_var = model_spec.y_var
    y_var_spec = dataset_spec.get_var_spec(y_var)
    y_var_type = y_var_spec.var_type
    is_classifier = y_var_type in ['categorical', 'ordinal']
    
    # Run cross-validation for each fold
    for fold_idx in range(cv_spec.n_splits):
        # Check if fold results already exist and should be loaded
        fold_results_path = None
        if output_folder:
            fold_model_path = os.path.join(
                output_folder, f"fold_{fold_idx}_model.pkl"
            )
            fold_results_path = os.path.join(
                output_folder, f"fold_{fold_idx}_results.csv"
            )
            
            # If files exist and we're not overwriting or rerunning, load results
            if (os.path.exists(fold_model_path) and
                os.path.exists(fold_results_path) and 
                not overwrite_files and 
                not rerun_folds):
                try:
                    # Load the model
                    with open(fold_model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Load the results from CSV
                    fold_df = pd.read_csv(fold_results_path)
                    
                    print(f"Loaded existing results for fold {fold_idx}")
                    
                    # Append to our observation results
                    observation_results.append(fold_df)
                    
                    # Create a minimal fold result object with just the model
                    fold_result = {
                        'model': model,
                        'fold_idx': fold_idx
                    }
                    
                    fold_results.append(fold_result)
                    continue
                except Exception as e:
                    print(
                        f"Error loading fold {fold_idx} results: {e}. "
                        "Will recompute."
                    )
        
        # Process this fold
        try:
            print(f"Processing fold {fold_idx + 1} of {cv_spec.n_splits}...")
            
            # Get train and test indices for the specific fold
            train_indices, test_indices = cv_spec.get_fold_indices(
                fold_idx, len(dataframe)
            )
            
            # Split the dataframe
            train_df = dataframe.iloc[train_indices]
            test_df = dataframe.iloc[test_indices]
            
            # Convert train and test dataframes to MixedDataset objects
            train_dataset = dataframe_to_mixed_dataset(
                train_df, dataset_spec, model_spec
            )
            test_dataset = dataframe_to_mixed_dataset(
                test_df, dataset_spec, model_spec
            )
            
            # Train the model
            model = _fit_model(train_dataset, model_spec, random_seed)
            
            # Process test (out-of-sample) observations
            test_results = _process_fold_observations(
                model,
                test_dataset,
                test_indices,
                fold_idx,
                'test',
                dataframe.index,
                is_classifier
            )
            
            # Process train (in-sample) observations
            train_results = _process_fold_observations(
                model,
                train_dataset,
                train_indices,
                fold_idx,
                'train',
                dataframe.index,
                is_classifier
            )
            
            # Combine results
            fold_obs_results = test_results + train_results
            
            # Convert fold results to DataFrame
            fold_df = pd.DataFrame(fold_obs_results)
            
            # Append to overall observation results
            observation_results.append(fold_df)
            
            # Store fold result with model
            fold_result = {
                'model': model,
                'fold_idx': fold_idx
            }
            
            fold_results.append(fold_result)
            
            # Save results if an output folder is provided
            if output_folder:
                # Create output folder if it doesn't exist
                os.makedirs(output_folder, exist_ok=True)
                
                # Save the trained model
                with open(fold_model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Save fold results as CSV
                fold_df.to_csv(fold_results_path, index=False)
                print(f"Saved model for fold {fold_idx} to {fold_model_path}")
                print(
                    f"Saved results for fold {fold_idx} to {fold_results_path}"
                )
            
        except Exception as e:
            print(f"Error in fold {fold_idx}: {str(e)}")
            # Continue with next fold instead of failing completely
            continue
    
    # Combine all observation results into a single DataFrame
    all_observations_df = pd.concat(
        observation_results, ignore_index=True
    ) if observation_results else pd.DataFrame()
    
    # Calculate overall statistics
    results_summary = {}
    
    if not all_observations_df.empty:
        # Calculate test (out-of-sample) statistics
        test_obs = all_observations_df[all_observations_df['sample_type'] == 'test']
        if not test_obs.empty:
            test_avg_loss = test_obs['loss'].mean()
            results_summary['test_avg_loss'] = test_avg_loss
        
        # Calculate train (in-sample) statistics
        train_obs = all_observations_df[all_observations_df['sample_type'] == 'train']
        if not train_obs.empty:
            train_avg_loss = train_obs['loss'].mean()
            results_summary['train_avg_loss'] = train_avg_loss
        
        # Determine what type of loss metric was used
        loss_type = "log_loss" if is_classifier else "mean_squared_error"
        results_summary['loss_type'] = loss_type
    
    # Add completion statistics
    results_summary['n_folds_completed'] = len(fold_results)
    results_summary['n_folds_total'] = cv_spec.n_splits
    
    # Save overall results if an output folder is provided
    if output_folder and not all_observations_df.empty:
        # Save all observations as CSV
        all_obs_path = os.path.join(output_folder, "all_observations.csv")
        all_observations_df.to_csv(all_obs_path, index=False)
        print(
            f"Saved all observations with in-sample and out-of-sample results "
            f"to {all_obs_path}"
        )
    
    # Create the final CV summary dictionary
    cv_summary = {
        'fold_results': fold_results,
        'observation_results': all_observations_df,
        **results_summary
    }
    
    return cv_summary


def _validate_cv_inputs(
    dataframe: 'DataFrame', 
    dataset_spec: 'DatasetSpec', 
    cv_spec: 'CrossValidationFoldsSpec', 
    model_spec: 'SingleTargetModelSpec'
) -> None:
    """Validate inputs for cross-validation.
    
    Args:
        dataframe: The data containing all variables.
        dataset_spec: The specification describing the variables.
        cv_spec: The specification for cross-validation folds.
        model_spec: The model specification.
        
    Raises:
        ValueError: If inputs are invalid.
    """
    # Validate that the dataframe contains all required columns
    required_columns = list(dataset_spec.all_var_names)
    missing_columns = [col for col in required_columns 
                      if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            f"Dataframe is missing required columns: {missing_columns}"
        )
    
    # Check that cv_spec is valid
    if not hasattr(cv_spec, 'n_splits') or not hasattr(cv_spec,
                                                       'create_folds'):
        raise ValueError(
            "cv_spec must be a CrossValidationFoldsSpec object with n_splits "
            "and create_folds"
        )
    
    # Check that model_spec is compatible with dataset_spec
    if not hasattr(model_spec, 'y_var') or not hasattr(model_spec,
                                                       'independent_vars'):
        raise ValueError(
            "model_spec must be a SingleTargetModelSpec object with y_var "
            "and independent_vars"
        )
    
    # Check that y_var exists in dataset_spec
    if model_spec.y_var not in dataset_spec.all_var_names:
        raise ValueError(
            f"y_var '{model_spec.y_var}' not found in dataset_spec"
        )
    
    # Check that all independent variables exist in dataset_spec
    missing_vars = [var for var in model_spec.independent_vars 
                   if var not in dataset_spec.all_var_names]
    if missing_vars:
        raise ValueError(
            "The following independent variables are not found in "
            f"dataset_spec: {missing_vars}"
        )
