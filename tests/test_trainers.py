"""Tests for training implementation functions."""
import numpy as np
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from mixalot.datasets import DatasetSpec, MixedDataset, VarSpec
from mixalot.models import RandomForestSpec
from mixalot.trainers import train_random_forest


@pytest.fixture
def cat_dataset_spec():
    """Create a dataset specification with a categorical target variable."""
    cat_var_specs = [
        VarSpec('cat_var1', 'categorical', 
                categorical_mapping=[{'yes'}, {'no'}]),
        VarSpec('cat_var2', 'categorical', 
                categorical_mapping=[{'a'}, {'b'}, {'c'}])
    ]
    
    ord_var_specs = [
        VarSpec('ord_var1', 'ordinal', 
                categorical_mapping=[{'low'}, {'medium'}, {'high'}])
    ]
    
    num_var_specs = [
        VarSpec('num_var1', 'numerical'),
        VarSpec('num_var2', 'numerical')
    ]
    
    return DatasetSpec(cat_var_specs, ord_var_specs, num_var_specs)


@pytest.fixture
def num_dataset_spec():
    """Create a dataset specification with a numerical target variable."""
    cat_var_specs = [
        VarSpec('cat_var1', 'categorical', 
                categorical_mapping=[{'yes'}, {'no'}])
    ]
    
    ord_var_specs = [
        VarSpec('ord_var1', 'ordinal', 
                categorical_mapping=[{'low'}, {'medium'}, {'high'}])
    ]
    
    num_var_specs = [
        VarSpec('num_var1', 'numerical'),
        VarSpec('num_var2', 'numerical'),
        VarSpec('target', 'numerical')
    ]
    
    return DatasetSpec(cat_var_specs, ord_var_specs, num_var_specs)


@pytest.fixture
def cat_rf_spec():
    """Create a random forest spec for categorical prediction."""
    return RandomForestSpec(
        y_var='cat_var1',
        independent_vars=['cat_var2', 'ord_var1', 'num_var1', 'num_var2'],
        hyperparameters={
            'n_estimators': 10,
            'max_features': {'type': 'float', 'value': 0.7}
        }
    )


@pytest.fixture
def num_rf_spec():
    """Create a random forest spec for numerical prediction."""
    return RandomForestSpec(
        y_var='target',
        independent_vars=['cat_var1', 'ord_var1', 'num_var1', 'num_var2'],
        hyperparameters={
            'n_estimators': 10,
            'max_features': {'type': 'int', 'value': 2}
        }
    )


@pytest.fixture
def cat_dataset(cat_dataset_spec, cat_rf_spec):
    """Create a mixed dataset with categorical target."""
    # Generate random data for categorical variables
    # Categories are 1-indexed with 0 reserved for missing values
    num_samples = 100
    Xcat = np.random.randint(1, 3, size=(num_samples, 2))  # 2 cat variables
    Xord = np.random.randint(1, 4, size=(num_samples, 1))  # 1 ord variable
    Xnum = np.random.randn(num_samples, 2)  # 2 num variables
    
    # Create dataset
    return MixedDataset(
        dataset_spec=cat_dataset_spec,
        Xcat=Xcat,
        Xord=Xord,
        Xnum=Xnum,
        model_spec=cat_rf_spec
    )


@pytest.fixture
def num_dataset(num_dataset_spec, num_rf_spec):
    """Create a mixed dataset with numerical target."""
    # Generate random data
    num_samples = 100
    Xcat = np.random.randint(1, 3, size=(num_samples, 1))  # 1 cat variable
    Xord = np.random.randint(1, 4, size=(num_samples, 1))  # 1 ord variable
    Xnum = np.random.randn(num_samples, 3)  # 3 num variables (w/ target)
    
    # Create dataset
    return MixedDataset(
        dataset_spec=num_dataset_spec,
        Xcat=Xcat,
        Xord=Xord,
        Xnum=Xnum,
        model_spec=num_rf_spec
    )


@pytest.fixture
def dataset_with_missing_cat(cat_dataset_spec, cat_rf_spec):
    """Create a dataset with missing categorical values."""
    num_samples = 100
    
    # Create data with missing values (0 for categorical)
    Xcat = np.random.randint(1, 3, size=(num_samples, 2))
    # Introduce some missing values
    Xcat[0, 0] = 0
    Xcat[0, 1] = 0
    Xcat[1, 0] = 0
    Xord = np.random.randint(1, 4, size=(num_samples, 1))
    Xnum = np.random.randn(num_samples, 2)
    
    # Create dataset
    return MixedDataset(
        dataset_spec=cat_dataset_spec,
        Xcat=Xcat,
        Xord=Xord,
        Xnum=Xnum,
        model_spec=cat_rf_spec
    )


@pytest.fixture
def dataset_with_missing_ord(cat_dataset_spec, cat_rf_spec):
    """Create a dataset with missing ordinal values."""
    num_samples = 100
    
    # Create data with missing values (0 for ordinal)
    Xcat = np.random.randint(1, 3, size=(num_samples, 2))
    Xord = np.random.randint(1, 4, size=(num_samples, 1))
    # Introduce a missing value
    Xord[0, 0] = 0
    Xnum = np.random.randn(num_samples, 2)
    
    # Create dataset
    return MixedDataset(
        dataset_spec=cat_dataset_spec,
        Xcat=Xcat,
        Xord=Xord,
        Xnum=Xnum,
        model_spec=cat_rf_spec
    )


@pytest.fixture
def dataset_with_missing_num(cat_dataset_spec, cat_rf_spec):
    """Create a dataset with missing numerical values."""
    num_samples = 100
    
    # Create data with missing values (NaN for numerical)
    Xcat = np.random.randint(1, 3, size=(num_samples, 2))
    Xord = np.random.randint(1, 4, size=(num_samples, 1))
    Xnum = np.random.randn(num_samples, 2)
    # Introduce some missing values
    Xnum[0, 0] = np.nan
    Xnum[0, 1] = np.nan
    Xnum[1, 0] = np.nan
    
    # Create dataset
    return MixedDataset(
        dataset_spec=cat_dataset_spec,
        Xcat=Xcat,
        Xord=Xord,
        Xnum=Xnum,
        model_spec=cat_rf_spec
    )


class TestTrainRandomForest():
    """Tests for train_random_forest function."""
    
    def test_train_categorical_model(self, cat_dataset, cat_rf_spec):
        """Test training a categorical model."""
        model = train_random_forest(cat_dataset, cat_rf_spec, random_seed=42)
        
        # Check that the model is of the correct type
        assert isinstance(model, RandomForestClassifier)
        
        # Check that the model has the expected number of estimators
        assert model.n_estimators == 10
        
        # Check that the model has feature importances
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == 4  # 4 independent variables

    def test_train_numerical_model(self, num_dataset, num_rf_spec):
        """Test training a numerical model."""
        model = train_random_forest(num_dataset, num_rf_spec, random_seed=42)
        
        # Check that the model is of the correct type
        assert isinstance(model, RandomForestRegressor)
        
        # Check that the model has the expected number of estimators
        assert model.n_estimators == 10
        
        # Check that the model has feature importances
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == 4  # 4 independent variables

    def test_error_with_missing_categorical_data(
            self, dataset_with_missing_cat, cat_rf_spec):
        """Test that an error is raised when categorical data is missing."""
        print(dataset_with_missing_cat.Xcat)
        with pytest.raises(ValueError) as excinfo:
            train_random_forest(dataset_with_missing_cat, cat_rf_spec)
        
        assert "Missing values detected in categorical variables"\
            in str(excinfo.value)

    def test_error_with_missing_ordinal_data(
            self, dataset_with_missing_ord, cat_rf_spec):
        """Test that an error is raised when ordinal data is missing."""
        with pytest.raises(ValueError) as excinfo:
            train_random_forest(dataset_with_missing_ord, cat_rf_spec)
        
        assert "Missing values detected in ordinal variables"\
            in str(excinfo.value)

    def test_error_with_missing_numerical_data(
            self, dataset_with_missing_num, cat_rf_spec):
        """Test that an error is raised when numerical data is missing."""
        with pytest.raises(ValueError) as excinfo:
            train_random_forest(dataset_with_missing_num, cat_rf_spec)
        
        assert "Missing values detected in numerical variables"\
            in str(excinfo.value)

    def test_prediction_with_trained_model(self, cat_dataset, cat_rf_spec):
        """Test making predictions with a trained model."""
        model = train_random_forest(cat_dataset, cat_rf_spec, random_seed=42)
        
        # Get one sample from the dataset
        Xcat, Xord, Xnum, _ = cat_dataset.get_arrays()
        X_combined = torch.cat(
            [tensor.float() for tensor in (Xcat, Xord, Xnum) 
             if tensor is not None], 
            dim=1
        )
        X_sample = X_combined[0:1].cpu().numpy()
        
        # Make a prediction
        prediction = model.predict(X_sample)
        
        # Check that the prediction is valid (either 0 or 1 for binary
        # classification)
        assert prediction[0] in [1, 2]
    
    def test_error_with_missing_y_data(self, cat_dataset_spec):
        """Test that an error is raised when y_data is missing."""
        # Create dataset without model_spec so y_data will be None
        num_samples = 50
        Xcat = np.random.randint(1, 3, size=(num_samples, 2))
        Xord = np.random.randint(1, 4, size=(num_samples, 1))
        Xnum = np.random.randn(num_samples, 2)
        
        dataset = MixedDataset(
            dataset_spec=cat_dataset_spec,
            Xcat=Xcat,
            Xord=Xord,
            Xnum=Xnum,
            # No model_spec provided
        )
        
        # Create a model spec
        rf_spec = RandomForestSpec(
            y_var='cat_var1',
            independent_vars=['cat_var2', 'ord_var1', 'num_var1'],
            hyperparameters={
                'n_estimators': 10,
                'max_features': {'type': 'float', 'value': 0.7}
            }
        )
        
        # Test that an error is raised
        with pytest.raises(ValueError) as excinfo:
            train_random_forest(dataset, rf_spec)
        
        assert "No target variable found in dataset" in str(excinfo.value)
