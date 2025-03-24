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
    
    This function runs cross-validation by calling fit_single_fold for each
    fold and aggregating the results. If an output folder is provided, the
    function will save the results for each fold to files in that folder.
    
    Args:
        dataframe (pd.DataFrame): The data containing all variables specified
            in dataset_spec.
        dataset_spec (DatasetSpec): The specification describing the variables
            in the dataset.
        cv_spec (CrossValidationFoldsSpec): The specification for
            cross-validation folds.
        model_spec (SingleTargetModelSpec): The model specification.
        random_seed (int, optional): Seed for reproducibility.
        output_folder (str, optional): Folder to save results to. If None,
            results are not saved to disk.
        overwrite_files (bool): If True, overwrites existing files in the
            output folder. Default is False.
        rerun_folds (bool): If True, reruns all folds even if results already
            exist. Default is False.
            
    Returns:
        dict: A dictionary containing:
            - 'fold_results': List of results from each fold
            - 'average_loss': Mean loss across all folds
            - 'std_loss': Standard deviation of loss across all folds
            - 'all_predictions': Dictionary mapping test indices to predicted
                values across all folds
    """
    # Validate input parameters
    _validate_cv_inputs(dataframe, dataset_spec, cv_spec, model_spec)
    
    # Create output folder if it doesn't exist
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Initialize containers for results
    fold_results = []
    all_predictions = {}
    
    # Get y variable type to determine loss type
    y_var = model_spec.y_var
    y_var_spec = dataset_spec.get_var_spec(y_var)
    y_var_type = y_var_spec.var_type
    is_classifier = y_var_type in ['categorical', 'ordinal']
    
    # Run cross-validation for each fold
    for fold_idx in range(cv_spec.n_splits):
        # Check if fold results already exist and should be loaded
        if output_folder:
            # Save model and fold info separately
            fold_model_path = os.path.join(output_folder,
                                           f"fold_{fold_idx}_model.pkl")
            fold_results_path = os.path.join(output_folder,
                                             f"fold_{fold_idx}_results.csv")
            
            # If files exist and we're not overwriting or rerunning, load the
            # existing results
            if (os.path.exists(fold_model_path) and
                os.path.exists(fold_results_path) and 
                not overwrite_files and 
                not rerun_folds):
                try:
                    # Load the model
                    with open(fold_model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Load the results from CSV
                    results_df = pd.read_csv(fold_results_path)
                    
                    # Reconstruct the predictions dictionary
                    predictions = {}
                    for _, row in results_df.iterrows():
                        idx = int(row['index'])
                        
                        # For classification, we need to reconstruct the
                        # probability array
                        if is_classifier:
                            # Find all probability columns
                            prob_columns = [col for col in results_df.columns
                                            if col.startswith('prob_')]
                            probs = [row[col] for col in prob_columns]
                            predictions[idx] = np.array(probs)
                        else:
                            # For regression, just extract the prediction
                            predictions[idx] = row['prediction']
                    
                    # Reconstruct the fold result
                    fold_result = {
                        'model': model,
                        'fold_loss': results_df['fold_loss'].iloc[0],
                        'test_indices': results_df['index'].values,
                        'predictions': predictions
                    }
                    
                    print(f"Loaded existing results for fold {fold_idx}")
                    fold_results.append(fold_result)
                    all_predictions.update(predictions)
                    continue
                except Exception as e:
                    print(
                        f"Error loading fold {fold_idx} results: {e}. Will "
                        "recompute."
                    )
        
        # Fit the model on this fold
        try:
            print(f"Fitting fold {fold_idx + 1} of {cv_spec.n_splits}...")
            fold_result = fit_single_fold(
                dataframe,
                dataset_spec,
                cv_spec,
                model_spec,
                fold_idx,
                random_seed
            )
            
            # Save the results if an output folder is provided
            if output_folder:
                # Create output folder if it doesn't exist
                os.makedirs(output_folder, exist_ok=True)
                
                # Save the trained model (can only be saved as pickle)
                with open(fold_model_path, 'wb') as f:
                    pickle.dump(fold_result['model'], f)
                
                # Create a DataFrame for CSV export
                results_data = []
                for idx in fold_result['test_indices']:
                    row_data = {'index': idx}
                    
                    # Add fold loss to each row for reference
                    row_data['fold_loss'] = fold_result['fold_loss']
                    
                    # Add predictions (different format for classification vs
                    # regression)
                    if is_classifier:
                        # For classification, we need to save the probability
                        # for each class
                        probs = fold_result['predictions'][int(idx)]
                        for i, prob in enumerate(probs):
                            row_data[f'prob_{i}'] = prob
                    else:
                        # For regression, just store the predicted value
                        row_data['prediction'] =\
                            fold_result['predictions'][int(idx)]
                    
                    results_data.append(row_data)
                
                # Save as CSV
                results_df = pd.DataFrame(results_data)
                results_df.to_csv(fold_results_path, index=False)
                
                print(
                    f"Saved model for fold {fold_idx} to {fold_model_path}"
                )
                print(
                    f"Saved results for fold {fold_idx} to {fold_results_path}"
                )
            
            # Append fold results and update all predictions
            fold_results.append(fold_result)
            all_predictions.update(fold_result['predictions'])
            
        except Exception as e:
            print(f"Error in fold {fold_idx}: {str(e)}")
            # Continue with next fold instead of failing completely
            continue
    
    losses = [result['fold_loss'] for result in fold_results]
    average_loss = np.mean(losses)
    std_loss = np.std(losses)
        
    # Determine what type of loss metric was used
    loss_type = "log_loss" if is_classifier else "mean_squared_error"
        
    # Create summary dictionary
    cv_summary = {
        'fold_results': fold_results,
        'average_loss': average_loss,
        'std_loss': std_loss,
        'all_predictions': all_predictions,
        'loss_type': loss_type,
        'n_folds_completed': len(fold_results),
        'n_folds_total': cv_spec.n_splits
    }
        
    # Save overall summary if an output folder is provided
    if output_folder:
        # Save summary as CSV for better human readability
        summary_path = os.path.join(output_folder, "cv_summary.csv")
            
        # Create a DataFrame for the summary
        summary_data = {
            'fold': list(range(len(losses))) + ['average', 'std_dev'],
            'loss': losses + [average_loss, std_loss],
            'loss_type': [loss_type] * (len(losses) + 2),
        }
            
        # Add CV parameters for reference
        summary_data['cv_n_splits'] =\
            [cv_spec.n_splits] * (len(losses) + 2)
        summary_data['cv_random_state'] =\
            [cv_spec.random_state] * (len(losses) + 2)
        summary_data['target_variable'] =\
            [model_spec.y_var] * (len(losses) + 2)
            
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False)
            
        # Also save a comprehensive predictions file
        all_preds_path = os.path.join(output_folder, "all_predictions.csv")
            
        # Create a DataFrame with all predictions
        preds_data = []
        for idx, pred in all_predictions.items():
            row_data = {'index': idx}
                
            # Format differently based on classification vs regression
            if is_classifier:
                for i, prob in enumerate(pred):
                    row_data[f'prob_{i}'] = prob
            else:
                row_data['prediction'] = pred
            
            preds_data.append(row_data)
         
        if preds_data:  # Check if we have predictions to save
            preds_df = pd.DataFrame(preds_data)
            preds_df.to_csv(all_preds_path, index=False)
            print(f"Saved all predictions to {all_preds_path}")
        
        print(f"Saved cross-validation summary to {summary_path}")
    
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
