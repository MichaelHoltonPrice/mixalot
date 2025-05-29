"""Tests for device handling across CPU and GPU."""
import numpy as np
import pandas as pd
import torch

from mixalot.crossval import _calculate_losses, dataframe_to_mixed_dataset
from mixalot.datasets import DatasetSpec, VarSpec
from mixalot.models import ANNEnsembleSpec
from mixalot.trainers import BasicAnn, EnsembleTorchModel, train_ann_ensemble


class TestDeviceAvailability:
    """Test that both CPU and GPU are available and working."""
    
    def test_cuda_is_available(self):
        """Test that CUDA is available on this system."""
        assert torch.cuda.is_available(),\
            "CUDA must be available for these tests"
    
    def test_cpu_and_gpu_tensors(self):
        """Test that we can create tensors on both CPU and GPU."""
        # Create CPU tensor
        cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
        assert cpu_tensor.device.type == 'cpu'
        
        # Create GPU tensor
        gpu_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        assert gpu_tensor.device.type == 'cuda'
        
        # Move between devices
        moved_to_gpu = cpu_tensor.cuda()
        assert moved_to_gpu.device.type == 'cuda'
        
        moved_to_cpu = gpu_tensor.cpu()
        assert moved_to_cpu.device.type == 'cpu'
    
    def test_ann_model_trains_on_cuda(self):
        """Test that ANN models are trained on CUDA by default."""
        # Create simple dataset spec
        cat_var_spec = VarSpec(
            var_name='y',
            var_type='categorical',
            categorical_mapping=[{'a'}, {'b'}]
        )
        num_var_spec = VarSpec(
            var_name='x',
            var_type='numerical'
        )
        dataset_spec = DatasetSpec(
            cat_var_specs=[cat_var_spec],
            ord_var_specs=[],
            num_var_specs=[num_var_spec]
        )
        
        # Create simple model spec
        model_spec = ANNEnsembleSpec(
            y_var='y',
            independent_vars=['x'],
            hyperparameters={
                'hidden_sizes': [4],
                'dropout_prob': 0.1,
                'num_models': 1,
                'epochs': 2
            }
        )
        
        # Create simple data
        df = pd.DataFrame({
            'y': ['a', 'b'] * 10,
            'x': np.random.randn(20)
        })
        
        # Convert to dataset
        dataset = dataframe_to_mixed_dataset(df, dataset_spec, model_spec)
        
        # Train model
        model = train_ann_ensemble(dataset, model_spec, random_seed=42)
        
        # Check that model is on CUDA
        for ann in model.models:
            for param in ann.parameters():
                assert param.is_cuda, "Model should be trained on CUDA"
    
    def test_calculate_losses_handles_cuda_model(self):
        """Test that _calculate_losses works with CUDA models."""
        # Create simple ensemble on CUDA
        ensemble = EnsembleTorchModel(
            num_models=1,
            lr=0.001,
            base_model_class=BasicAnn,
            num_x_var=2,
            num_cat=2,
            hidden_sizes=[4],
            dropout_prob=0.1
        )
        
        # Move to CUDA
        for model in ensemble.models:
            model.cuda()
        
        # CPU features (numpy)
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        true_values = np.array([1, 2])
        
        # This should work - _calculate_losses handles device transfer
        predictions, losses = _calculate_losses(
            ensemble, features, true_values, is_classifier=True
        )
        
        # Verify we got results
        assert len(predictions) == 2
        assert len(losses) == 2
