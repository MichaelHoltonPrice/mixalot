"""Tests for training implementation functions."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from mixalot.datasets import AnnEnsembleDataset, DatasetSpec, MixedDataset, VarSpec
from mixalot.models import ANNEnsembleSpec, RandomForestSpec
from mixalot.trainers import (
    BasicAnn,
    EnsembleTorchModel,
    train_ann_ensemble,
    train_random_forest
)


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
def ord_dataset(cat_dataset_spec, ord_ann_spec):
    """Create a mixed dataset with ordinal target."""
    # Generate random data
    num_samples = 64  # Multiple of batch size for easier testing
    Xcat = np.random.randint(1, 3, size=(num_samples, 2))  # 2 cat variables
    Xord = np.random.randint(1, 4, size=(num_samples, 1))  # 1 ord variable
    Xnum = np.random.randn(num_samples, 2)  # 2 num variables
    
    # Create dataset
    return MixedDataset(
        dataset_spec=cat_dataset_spec,
        Xcat=Xcat,
        Xord=Xord,
        Xnum=Xnum,
        model_spec=ord_ann_spec
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


@pytest.fixture
def cat_ann_spec():
    """Create an ANN ensemble spec for categorical prediction."""
    return ANNEnsembleSpec(
        y_var='cat_var1',
        independent_vars=['cat_var2', 'ord_var1', 'num_var1', 'num_var2'],
        hyperparameters={
            'hidden_sizes': [8, 4],
            'dropout_prob': 0.5,
            'num_models': 2,
            'batch_size': 16,
            'lr': 0.01,
            'final_lr': 0.001,
            'epochs': 5  # Small number for faster tests
        }
    )


@pytest.fixture
def ord_ann_spec():
    """Create an ANN ensemble spec for ordinal prediction."""
    return ANNEnsembleSpec(
        y_var='ord_var1',
        independent_vars=['cat_var1', 'cat_var2', 'num_var1', 'num_var2'],
        hyperparameters={
            'hidden_sizes': [8, 4],
            'dropout_prob': 0.5,
            'num_models': 2,
            'batch_size': 16,
            'lr': 0.01,
            'final_lr': 0.001,
            'epochs': 5  # Small number for faster tests
        }
    )


@pytest.fixture
def controlled_cat_dataset(cat_dataset_spec, cat_ann_spec):
    """Create a mixed dataset with categorical target and controlled values."""
    # Create arrays with explicitly set values
    num_samples = 64
    
    # Create categorical data - 2 variables
    Xcat = np.ones((num_samples, 2), dtype=int)  # All ones initially
    # Set some specific values
    Xcat[0:20, 0] = 1  # cat_var1 (target) - first 20 samples are class 1
    Xcat[20:40, 0] = 2  # cat_var1 (target) - next 20 samples are class 2
    Xcat[40:60, 0] = 1  # cat_var1 (target) - next 20 samples are class 1
    Xcat[60:64, 0] = 2  # cat_var1 (target) - last 4 samples are class 2
    
    Xcat[:, 1] = 1  # cat_var2 - all samples are class 1
    Xcat[10:30, 1] = 2  # Set some to class 2
    Xcat[40:50, 1] = 3  # Set some to class 3
    
    # Create ordinal data - 1 variable
    Xord = np.ones((num_samples, 1), dtype=int)  # All ones initially
    Xord[15:35, 0] = 2  # Set some to medium
    Xord[45:60, 0] = 3  # Set some to high
    
    # Create numerical data - 2 variables
    Xnum = np.zeros((num_samples, 2), dtype=float)  # All zeros initially
    Xnum[:, 0] = np.linspace(0, 1, num_samples)  # First variable varies from 0 to 1
    Xnum[:, 1] = np.sin(np.linspace(0, 3*np.pi, num_samples))  # Second variable is a sine wave
    
    return MixedDataset(
        dataset_spec=cat_dataset_spec,
        Xcat=Xcat,
        Xord=Xord,
        Xnum=Xnum,
        model_spec=cat_ann_spec
    )


@pytest.fixture
def controlled_ord_dataset(cat_dataset_spec, ord_ann_spec):
    """Create a mixed dataset with ordinal target and controlled values."""
    num_samples = 64
    
    # Create categorical data - 2 variables
    Xcat = np.ones((num_samples, 2), dtype=int)  # All ones initially
    Xcat[:, 0] = 1  # cat_var1 - all samples are class 1
    Xcat[10:30, 0] = 2  # Set some to class 2
    
    Xcat[:, 1] = 1  # cat_var2 - all samples are class 1
    Xcat[20:40, 1] = 2  # Set some to class 2
    Xcat[50:60, 1] = 3  # Set some to class 3
    
    # Create ordinal data - 1 variable
    Xord = np.ones((num_samples, 1), dtype=int)  # All ones initially (low)
    Xord[0:20, 0] = 1  # ord_var1 (target) - first 20 samples are low
    Xord[20:40, 0] = 2  # ord_var1 (target) - next 20 samples are medium
    Xord[40:64, 0] = 3  # ord_var1 (target) - last 24 samples are high
    
    # Create numerical data - 2 variables
    Xnum = np.zeros((num_samples, 2), dtype=float)  # All zeros initially
    Xnum[:, 0] = np.linspace(-1, 1, num_samples)  # First variable varies from -1 to 1
    Xnum[:, 1] = np.cos(np.linspace(0, 2*np.pi, num_samples))  # Second variable is a cosine wave
    
    return MixedDataset(
        dataset_spec=cat_dataset_spec,
        Xcat=Xcat,
        Xord=Xord,
        Xnum=Xnum,
        model_spec=ord_ann_spec
    )


@pytest.fixture
def dataset_with_missing_x_values(cat_dataset_spec, cat_ann_spec):
    """Create a dataset with missing values in X variables but not target."""
    num_samples = 64
    
    # Create categorical data - 2 variables
    Xcat = np.ones((num_samples, 2), dtype=int)  # All ones initially
    
    # Target variable (no missing values)
    Xcat[0:32, 0] = 1  # cat_var1 (target) - first half are class 1
    Xcat[32:64, 0] = 2  # cat_var1 (target) - second half are class 2
    
    # Second categorical variable (with missing values)
    Xcat[:, 1] = 1  # cat_var2 - all samples are class 1
    Xcat[5:15, 1] = 2  # Set some to class 2
    Xcat[40:50, 1] = 3  # Set some to class 3
    Xcat[20:25, 1] = 0  # Set some values as missing
    
    # Create ordinal data - 1 variable (with missing values)
    Xord = np.ones((num_samples, 1), dtype=int)
    Xord[10:30, 0] = 2  # Set some to medium
    Xord[45:60, 0] = 3  # Set some to high
    Xord[35:40, 0] = 0  # Set some values as missing
    
    # Create numerical data - 2 variables (with missing values)
    Xnum = np.zeros((num_samples, 2), dtype=float)
    Xnum[:, 0] = np.linspace(0, 1, num_samples)
    Xnum[:, 1] = np.sin(np.linspace(0, 3*np.pi, num_samples))
    
    # Set some numerical values as missing (NaN)
    Xnum[15:20, 0] = np.nan
    Xnum[55:60, 1] = np.nan
    
    return MixedDataset(
        dataset_spec=cat_dataset_spec,
        Xcat=Xcat,
        Xord=Xord,
        Xnum=Xnum,
        model_spec=cat_ann_spec
    )


@pytest.fixture
def dataset_with_missing_target(cat_dataset_spec, cat_ann_spec):
    """Create a dataset with missing values in target variable."""
    num_samples = 64
    
    # Create categorical data - 2 variables
    Xcat = np.ones((num_samples, 2), dtype=int)
    
    # Target variable (with missing values)
    Xcat[0:30, 0] = 1  # cat_var1 (target) - first 30 are class 1
    Xcat[30:60, 0] = 2  # cat_var1 (target) - next 30 are class 2
    Xcat[60:64, 0] = 0  # cat_var1 (target) - last 4 are missing
    
    # Second categorical variable
    Xcat[:, 1] = 1  # cat_var2 - all samples are class 1
    Xcat[15:35, 1] = 2  # Set some to class 2
    
    # Create ordinal data - 1 variable
    Xord = np.ones((num_samples, 1), dtype=int)
    Xord[20:40, 0] = 2  # Set some to medium
    Xord[50:60, 0] = 3  # Set some to high
    
    # Create numerical data - 2 variables
    Xnum = np.zeros((num_samples, 2), dtype=float)
    Xnum[:, 0] = np.linspace(0, 1, num_samples)
    Xnum[:, 1] = np.sin(np.linspace(0, 3*np.pi, num_samples))
    
    return MixedDataset(
        dataset_spec=cat_dataset_spec,
        Xcat=Xcat,
        Xord=Xord,
        Xnum=Xnum,
        model_spec=cat_ann_spec
    )


class TestBasicAnn:
    """Tests for the BasicAnn neural network class."""
    
    def test_init_and_structure(self):
        """Test initialization and structure of the BasicAnn model."""
        model = BasicAnn(
            num_x_var=10,
            num_cat=3,
            hidden_sizes=[8, 4],
            dropout_prob=0.5
        )
        
        # Check model structure
        assert len(model.hidden_layers) == 4  # 2 hidden layers + 2 dropout
        assert isinstance(model.hidden_layers[0], torch.nn.Linear)
        assert model.hidden_layers[0].in_features == 10
        assert model.hidden_layers[0].out_features == 8
        
        assert isinstance(model.hidden_layers[1], torch.nn.Dropout)
        assert model.hidden_layers[1].p == 0.5
        
        assert isinstance(model.hidden_layers[2], torch.nn.Linear)
        assert model.hidden_layers[2].in_features == 8
        assert model.hidden_layers[2].out_features == 4
        
        assert isinstance(model.output_layer, torch.nn.Linear)
        assert model.output_layer.in_features == 4
        assert model.output_layer.out_features == 3
    
    def test_forward_pass(self):
        """Test forward pass of the BasicAnn model."""
        model = BasicAnn(
            num_x_var=10,
            num_cat=3,
            hidden_sizes=[8, 4],
            dropout_prob=0.5
        )
        
        # Test with a batch of 5 samples
        x = torch.randn(5, 10)
        outputs = model(x)
        
        # Check output shape
        assert outputs.shape == (5, 3)
        
        # Ensure output is not softmaxed (raw logits)
        # Raw logits can have values outside the [0,1] range
        assert torch.any(outputs > 1.0) or torch.any(outputs < 0.0)


class TestEnsembleTorchModel:
    """Tests for the EnsembleTorchModel class."""
    
    def test_init(self):
        """Test initialization of ensemble model."""
        ensemble = EnsembleTorchModel(
            num_models=3,
            lr=0.01,
            base_model_class=BasicAnn,
            num_x_var=10,
            num_cat=2,
            hidden_sizes=[8],
            dropout_prob=0.5
        )
        
        # Check that ensemble has expected number of models
        assert len(ensemble.models) == 3
        
        # Check that all models are BasicAnn instances
        assert all(isinstance(model, BasicAnn) for model in ensemble.models)
        
        # Check learning rate properties
        assert ensemble.lr == 0.01
        # Default to lr if final_lr not given
        assert ensemble.final_lr == 0.01
    
    def test_predict_prob(self):
        """Test prediction functionality of ensemble model."""
        ensemble = EnsembleTorchModel(
            num_models=3,
            lr=0.01,
            base_model_class=BasicAnn,
            num_x_var=10,
            num_cat=2,
            hidden_sizes=[8],
            dropout_prob=0.5
        )
        
        # Create a test input
        x = torch.randn(5, 10)  # 5 samples, 10 features
        device = torch.device('cpu')
        
        # Get predictions
        predictions = ensemble.predict_prob(x, device)
        
        # Check output shape and properties
        assert predictions.shape == (5, 2)  # 5 samples, 2 classes
        assert torch.all(predictions >= 0.0)  # Probabilities are non-negative
        assert torch.all(predictions <= 1.0)  # Probabilities are <= 1.0
        
        # Check that probabilities sum to approximately 1 for each sample
        assert torch.allclose(
            torch.sum(predictions, dim=1), 
            torch.ones(5), 
            rtol=1e-6
        )
    
    def test_train_and_evaluate(self):
        """Test training and evaluation of ensemble model."""
        # Create a simple synthetic dataset
        num_samples = 20
        num_features = 5
        num_classes = 3
        
        X = torch.randn(num_samples, num_features)
        y = torch.randint(0, num_classes, (num_samples,))
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=4)
        
        # Create ensemble
        ensemble = EnsembleTorchModel(
            num_models=2,
            lr=0.01,
            base_model_class=BasicAnn,
            num_x_var=num_features,
            num_cat=num_classes,
            hidden_sizes=[8],
            dropout_prob=0.5
        )
        
        # Train for a very short time
        device = torch.device('cpu')
        loss = ensemble.train(dataloader, device, epochs=2)
        
        # Verify loss is a number and not NaN
        assert isinstance(loss, float)
        assert not np.isnan(loss)
        
        # Test evaluation with a test dataset
        test_X = torch.randn(10, num_features)
        test_y = torch.randint(0, num_classes, (10,))
        test_dataset = TensorDataset(test_X, test_y)
        test_dataloader = DataLoader(
            test_dataset, batch_size=5
        )
        
        # Train with test dataset for evaluation
        train_loss, test_loss = ensemble.train(
            dataloader, device, epochs=1, test_dl=test_dataloader
        )
        
        # Verify both losses are numbers and not NaN
        assert isinstance(train_loss, float)
        assert isinstance(test_loss, float)
        assert not np.isnan(train_loss)
        assert not np.isnan(test_loss)


class TestTrainANNEnsemble:
    """Tests for train_ann_ensemble function."""
    
    def test_train_categorical_model(self, controlled_cat_dataset,
                                     cat_ann_spec):
        """Test training a categorical model with controlled data."""
        model = train_ann_ensemble(
            controlled_cat_dataset, cat_ann_spec, random_seed=42
        )
        
        # Check that the model is of the correct type
        assert isinstance(model, EnsembleTorchModel)
        
        # Check that the model has the expected number of sub-models
        assert len(model.models) == 2
        
        # Check that each model is a BasicAnn
        assert all(isinstance(m, BasicAnn) for m in model.models)
        
        # Verify the output size matches the number of classes in the spec
        first_model = model.models[0]
        assert first_model.output_layer.out_features == 2  # 2 classes

    def test_train_ordinal_model(self, controlled_ord_dataset, ord_ann_spec):
        """Test training an ordinal model with controlled data."""
        model = train_ann_ensemble(
            controlled_ord_dataset, ord_ann_spec, random_seed=42
        )
        
        # Check that the model is of the correct type
        assert isinstance(model, EnsembleTorchModel)
        
        # Check that the model has the expected number of sub-models
        assert len(model.models) == 2
        
        # Check that each model is a BasicAnn
        assert all(isinstance(m, BasicAnn) for m in model.models)
        
        # Verify the output size matches the number of classes in the spec
        first_model = model.models[0]
        assert first_model.output_layer.out_features == 3  # 3 classes

    def test_train_with_missing_x_values(
        self, dataset_with_missing_x_values, cat_ann_spec
    ):
        """Test training with missing values in X variables."""
        # This should work because ANN models can handle missing X values
        model = train_ann_ensemble(
            dataset_with_missing_x_values, cat_ann_spec, random_seed=42
        )
        
        # Check that the model is of the correct type and structure
        assert isinstance(model, EnsembleTorchModel)
        assert len(model.models) == 2
        assert all(isinstance(m, BasicAnn) for m in model.models)
        
        # Use the same AnnEnsembleDataset as used in training to format the
        # data
        ann_dataset = AnnEnsembleDataset(dataset_with_missing_x_values)
        
        # Get a sample with missing values (index 20 where cat_var2 is missing)
        features, _ = ann_dataset[20]
        
        # Add batch dimension
        features_batch = features.unsqueeze(0)
        
        # Should be able to predict without error
        # Use the same device that was used for training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        probs = model.predict_prob(features_batch, device)
        
        # Check prediction properties
        assert probs.shape == (1, 2)  # One sample, two classes
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)
        assert torch.allclose(
            torch.sum(probs, dim=1), 
            torch.ones(1, device=probs.device),  # ensure tensor on same device
            rtol=1e-6
        )

    def test_error_with_missing_target(
        self, dataset_with_missing_target, cat_ann_spec
    ):
        """Test that an error is raised with missing target values."""
        with pytest.raises(ValueError) as excinfo:
            train_ann_ensemble(dataset_with_missing_target, cat_ann_spec)
        
        assert "Missing values detected in target variable"\
            in str(excinfo.value)
        assert "does not support missing target values" in str(excinfo.value)

    def test_prediction_with_trained_model(
        self, controlled_cat_dataset, cat_ann_spec
    ):
        """Test making predictions with a trained model."""
        model = train_ann_ensemble(
            controlled_cat_dataset, cat_ann_spec, random_seed=42
        )
        
        # Get a sample using the same format as in training
        # Use the AnnEnsembleDataset wrapper to get the properly formatted
        # input
        ann_dataset = AnnEnsembleDataset(controlled_cat_dataset)
        X_sample, _ = ann_dataset[0]  # Get first sample features
        X_sample = X_sample.unsqueeze(0)  # Add batch dimension
        
        # Make a prediction
        # Use the same device that was used for training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        probs = model.predict_prob(X_sample, device)
        
        # Check prediction properties
        assert probs.shape == (1, 2)  # One sample, two classes
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)
        assert torch.allclose(
            torch.sum(probs, dim=1), 
            torch.ones(1, device=probs.device),  # make tensors on same device
            rtol=1e-6
        )
        
        # Test another sample
        X_sample2, _ = ann_dataset[20]  # Get sample from class 2
        X_sample2 = X_sample2.unsqueeze(0)  # Add batch dimension
        probs2 = model.predict_prob(X_sample2, device)
        
        # The predictions should be different for different classes
        assert not torch.allclose(probs, probs2)
   
    def test_error_with_numerical_target(self, cat_dataset_spec):
        """Test that an error is raised with a numerical target variable."""
        # Create a numerical target model spec
        num_ann_spec = ANNEnsembleSpec(
            y_var='num_var1',
            independent_vars=[
                'cat_var1', 'cat_var2', 'ord_var1', 'num_var2'
            ],
            hyperparameters={
                'hidden_sizes': [8, 4],
                'dropout_prob': 0.5,
                'num_models': 2,
                'epochs': 5
            }
        )
        
        # Create dataset
        num_samples = 64
        Xcat = np.random.randint(1, 3, size=(num_samples, 2))
        Xord = np.random.randint(1, 4, size=(num_samples, 1))
        Xnum = np.random.randn(num_samples, 2)
        
        dataset = MixedDataset(
            dataset_spec=cat_dataset_spec,
            Xcat=Xcat,
            Xord=Xord,
            Xnum=Xnum,
            model_spec=num_ann_spec
        )
        
        # Test that an error is raised
        with pytest.raises(ValueError) as excinfo:
            train_ann_ensemble(dataset, num_ann_spec)
        
        assert "only supports categorical or ordinal targets" \
               in str(excinfo.value)
    
    def test_error_with_missing_y_data(self, cat_dataset_spec):
        """Test that an error is raised when y_data is missing."""
        # Create dataset without model_spec so y_data will be None
        num_samples = 64
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
        ann_spec = ANNEnsembleSpec(
            y_var='cat_var1',
            independent_vars=['cat_var2', 'ord_var1', 'num_var1'],
            hyperparameters={
                'hidden_sizes': [8, 4],
                'dropout_prob': 0.5,
                'num_models': 2,
                'epochs': 5
            }
        )
        
        # Test that an error is raised
        with pytest.raises(ValueError) as excinfo:
            train_ann_ensemble(dataset, ann_spec)
        
        assert "No target variable found in dataset" in str(excinfo.value)
