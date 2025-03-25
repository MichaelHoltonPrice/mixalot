"""Training implementation for Random Forest models."""
from __future__ import annotations
from typing import Optional, Union

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from mixalot.datasets import AnnEnsembleDataset, MixedDataset
from mixalot.models import ANNEnsembleSpec, RandomForestSpec


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


class BasicAnn(nn.Module):
    """Basic artificial neural network (ANN) model.

    This class represents a basic ANN model with an arbitrary number of hidden 
    layers. The structure of the model includes an input layer, a sequence of 
    alternating dense and dropout layers, and an output layer.

    Args:
        num_x_var: The number of input variables.
        num_cat: The number of output categories.
        hidden_sizes: A list of the sizes of the hidden layers.
        dropout_prob: The dropout probability for the dropout layers.
    """

    def __init__(self, num_x_var, num_cat, hidden_sizes, dropout_prob):
        """Initialize a new BasicAnn instance."""
        super(BasicAnn, self).__init__()

        self.hidden_layers = nn.ModuleList()
        
        # Define the input size to the first layer
        input_size = num_x_var

        # Create the hidden layers
        for hidden_size in hidden_sizes:
            # Dense layer
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            # Dropout layer
            self.hidden_layers.append(nn.Dropout(dropout_prob))

            # The output size of the current layer is the input size of the
            # next
            input_size = hidden_size

        # Define the output layer
        self.output_layer = nn.Linear(input_size, num_cat)

    def forward(self, x):
        """Implement the forward pass.

        Args:
            x: The input tensor.

        Returns:
            The output tensor of the model.
        """
        # Pass through each hidden layer
        for i, hidden_layer in enumerate(self.hidden_layers):
            # Apply activation function after each dense layer
            if i % 2 == 0:  # Dense layers are at even indices
                x = hidden_layer(x)
                x = F.relu(x)
            else:  # Dropout layers are at odd indices
                x = hidden_layer(x)
            
        # Pass through the output layer
        # No activation function after the output layer for raw logits
        x = self.output_layer(x)

        return x


class EnsembleTorchModel:
    """An ensemble of PyTorch neural network models.

    This class handles the training of an ensemble of PyTorch models and 
    prediction using the ensemble. During prediction, the ensemble model 
    outputs the averaged probabilities from all individual models.

    Args:
        num_models: The number of models in the ensemble.
        lr: Learning rate for training the models.
        base_model_class: The class of the models forming the ensemble.
        final_lr: The final learning rate for the LambdaLR scheduler.
        base_model_args: Positional arguments to be passed to base_model_class.
        base_model_kwargs: Keyword arguments to be passed to base_model_class.
    """

    def __init__(
        self,
        num_models,
        lr,
        base_model_class,
        *base_model_args,
        final_lr=None,
        **base_model_kwargs
    ):
        """Initialize an EnsembleTorchModel instance."""
        self.models = [
            base_model_class(*base_model_args, **base_model_kwargs)
            for _ in range(num_models)
        ]
        self.lr = lr
        self.final_lr = final_lr if final_lr is not None else lr

    def train(self, train_dl, device, epochs, test_dl=None):
        """Train the ensemble model.

        Args:
            train_dl: The DataLoader for training data.
            device: The device (CPU/GPU) to be used for training.
            epochs: The number of epochs for training.
            test_dl: The DataLoader for test data (optional).

        Returns:
            float: The mean training loss for the ensemble model.
            float: The mean test loss if test_dl is provided.
        """
        # Use cross entropy loss
        criterion = nn.CrossEntropyLoss()

        # Set up learning rate decay
        initial_lr = self.lr
        final_lr = self.final_lr
        
        # Lambda function for learning rate decay
        # Linearly decrease from initial_lr to final_lr over epochs
        lambda1 = lambda epoch: (final_lr / initial_lr) + (
            (1 - epoch / epochs) * (1 - final_lr / initial_lr)
        )

        # Initialize losses
        ensemble_train_loss = 0.0
        ensemble_test_loss = 0.0 if test_dl else None

        # Train each model in the ensemble
        for i, model in enumerate(self.models):
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
            scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
            
            # Train the model for the specified number of epochs
            for epoch in range(epochs):
                # Training phase
                model.train()
                epoch_loss = 0.0
                batch_count = 0
                
                for inputs, targets in train_dl:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    
                    # Accumulate loss
                    epoch_loss += loss.item()
                    batch_count += 1
                
                # Update learning rate
                scheduler.step()
                
                # Calculate average loss for this epoch
                avg_epoch_loss =\
                    epoch_loss / batch_count if batch_count > 0 else 0
                
                # Optionally log progress
                if (epoch + 1) % (epochs // 10 or 1) == 0:
                    print(f"Model {i+1}, Epoch {epoch+1}/{epochs}, "
                          f"Loss: {avg_epoch_loss:.4f}")
            
            # Calculate training loss for this model
            train_loss = self._evaluate(model, train_dl, criterion, device)
            ensemble_train_loss += train_loss
            
            # Calculate test loss if test data is provided
            if test_dl:
                test_loss = self._evaluate(model, test_dl, criterion, device)
                ensemble_test_loss += test_loss
                print(f"Model {i+1} - Train Loss: {train_loss:.4f}, "
                      f"Test Loss: {test_loss:.4f}")
            else:
                print(f"Model {i+1} - Train Loss: {train_loss:.4f}")
        
        # Calculate average losses across all models
        ensemble_train_loss /= len(self.models)
        if test_dl:
            ensemble_test_loss /= len(self.models)
            return ensemble_train_loss, ensemble_test_loss
        
        return ensemble_train_loss

    def _evaluate(self, model, data_loader, criterion, device):
        """Evaluate a model on a dataset.

        Args:
            model: The model to evaluate.
            data_loader: The DataLoader containing the evaluation data.
            criterion: The loss function to use.
            device: The device to use for computation.

        Returns:
            float: The average loss on the dataset.
        """
        model.eval()
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                batch_count += 1
        
        return total_loss / batch_count if batch_count > 0 else 0

    def predict_prob(self, x, device):
        """Predict using the ensemble model.

        This method computes the averaged probabilities from all models.

        Args:
            x: The input tensor.
            device: The device (CPU/GPU) to be used for prediction.

        Returns:
            torch.Tensor: The tensor of averaged probabilities.
        """
        all_probabilities = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x.to(device))
                probs = F.softmax(logits, dim=1)
                all_probabilities.append(probs)

        # Stack predictions and compute the averaged probabilities
        stacked_probabilities = torch.stack(all_probabilities)
        average_probabilities = torch.mean(stacked_probabilities, dim=0)

        return average_probabilities


def train_ann_ensemble(
    dataset: MixedDataset,
    model_spec: ANNEnsembleSpec,
    random_seed: Optional[int] = None
) -> EnsembleTorchModel:
    """Train an ANN Ensemble model using the provided dataset and spec.

    This function extracts the training data from the MixedDataset based on
    the model specification and trains an ensemble of neural networks.
    Unlike Random Forest, ANN Ensemble can handle missing values in the
    independent variables, but not in the target variable.

    Args:
        dataset: The MixedDataset containing the data to train on.
        model_spec: The ANNEnsembleSpec containing model configuration.
        random_seed: Optional seed for reproducibility.

    Returns:
        The trained ANN Ensemble model.

    Raises:
        ValueError: If the target variable type is not supported or if
            the target variable contains missing values.
    """
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Get dataset specification and target variable information
    dataset_spec = dataset.dataset_spec
    y_var = model_spec.y_var
    y_var_spec = dataset_spec.get_var_spec(y_var)
    y_var_type = y_var_spec.var_type

    # Only support classification tasks for now
    if y_var_type not in ['categorical', 'ordinal']:
        raise ValueError(
            f"ANN Ensemble only supports categorical or ordinal targets, "
            f"got {y_var_type}"
        )

    # Extract data arrays to check target variable
    _, _, _, y_data = dataset.get_arrays()

    if y_data is None:
        raise ValueError("No target variable found in dataset")

    # Check for missing values in target variable (encoded as 0)
    if torch.any(y_data == 0):
        raise ValueError(
            "Missing values detected in target variable "
            f"'{model_spec.y_var}'. ANN Ensemble does not support missing "
            "target values."
        )
   
    # Create the wrapper dataset
    ann_dataset = AnnEnsembleDataset(dataset)
    
    # Create a DataLoader
    batch_size = model_spec.hyperparameters.get('batch_size', 32)
    dataloader = DataLoader(
        ann_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    # Get the number of features
    sample_features, _ = ann_dataset[0]
    num_features = len(sample_features)
    
    # Get the number of categories from the model spec
    if y_var_type == 'categorical':
        num_classes = len(y_var_spec.categorical_mapping)
    else:  # y_var_type == 'ordinal'
        num_classes = len(y_var_spec.categorical_mapping)
    
    # Extract hyperparameters for the ensemble
    hidden_sizes = model_spec.hyperparameters.get('hidden_sizes', [32, 16])
    dropout_prob = model_spec.hyperparameters.get('dropout_prob', 0.5)
    num_models = model_spec.hyperparameters.get('num_models', 5)
    learning_rate = model_spec.hyperparameters.get('lr', 0.001)
    final_lr = model_spec.hyperparameters.get('final_lr', learning_rate)
    epochs = model_spec.hyperparameters.get('epochs', 100)
    
    # Determine the device to use
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create the ensemble model
    ensemble = EnsembleTorchModel(
        num_models=num_models,
        lr=learning_rate,
        base_model_class=BasicAnn,
        num_x_var=num_features,
        num_cat=num_classes,
        hidden_sizes=hidden_sizes,
        dropout_prob=dropout_prob,
        final_lr=final_lr
    )
    
    # Train the ensemble
    ensemble.train(dataloader, device, epochs)
    
    return ensemble
