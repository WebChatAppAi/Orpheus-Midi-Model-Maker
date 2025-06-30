"""
Configuration Management Module
Handles YML configuration loading and validation
"""

import yaml
import os
import glob
from pathlib import Path

class ConfigManager:
    def __init__(self):
        self.required_fields = {
            'dataset': ['path'],
            'model': ['seq_len', 'dim', 'depth', 'heads'],
            'training': ['batch_size', 'gradient_accumulate_every', 'learning_rate', 'num_epochs'],
            'output': ['model_dir', 'sample_dir']
        }
    
    def load_config(self, config_path):
        """Load configuration from YML file"""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def validate_config(self, config):
        """Validate configuration structure and values"""
        errors = []
        
        # Check required sections
        for section, fields in self.required_fields.items():
            if section not in config:
                errors.append(f"Missing required section: {section}")
                continue
            
            # Check required fields in section
            for field in fields:
                if field not in config[section]:
                    errors.append(f"Missing required field: {section}.{field}")
        
        if errors:
            return {'valid': False, 'errors': errors}
        
        # Validate dataset path
        dataset_path = config['dataset']['path']
        if not self.validate_dataset_path(dataset_path):
            errors.append(f"Dataset path not found or contains no pickle files: {dataset_path}")
        
        # Validate model parameters
        model_params = config['model']
        if model_params['seq_len'] <= 0:
            errors.append("seq_len must be positive")
        if model_params['dim'] <= 0:
            errors.append("dim must be positive")
        if model_params['depth'] <= 0:
            errors.append("depth must be positive")
        if model_params['heads'] <= 0:
            errors.append("heads must be positive")
        
        # Validate training parameters
        training_params = config['training']
        if training_params['batch_size'] <= 0:
            errors.append("batch_size must be positive")
        if training_params['gradient_accumulate_every'] <= 0:
            errors.append("gradient_accumulate_every must be positive")
        if training_params['learning_rate'] <= 0:
            errors.append("learning_rate must be positive")
        if training_params['num_epochs'] <= 0:
            errors.append("num_epochs must be positive")
        
        # Validate output directories
        output_config = config['output']
        model_dir = Path(output_config['model_dir'])
        sample_dir = Path(output_config['sample_dir'])
        
        # Create directories if they don't exist
        model_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        return {'valid': len(errors) == 0, 'errors': errors}
    
    def validate_dataset_path(self, dataset_path):
        """Validate dataset path contains pickle files"""
        if not os.path.exists(dataset_path):
            return False
        
        # Find pickle files recursively
        pickle_files = []
        for ext in ['*.pkl', '*.pickle']:
            pattern = os.path.join(dataset_path, '**', ext)
            pickle_files.extend(glob.glob(pattern, recursive=True))
        
        return len(pickle_files) > 0
    
    def get_dataset_files(self, dataset_path):
        """Get all pickle files from dataset path"""
        pickle_files = []
        for ext in ['*.pkl', '*.pickle']:
            pattern = os.path.join(dataset_path, '**', ext)
            pickle_files.extend(glob.glob(pattern, recursive=True))
        
        return sorted(pickle_files)