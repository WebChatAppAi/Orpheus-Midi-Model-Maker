"""
Configuration Generator - Auto-generates optimal training configurations
"""

import os
import yaml
import pickle
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .utils import console, get_gpu_info, get_recommended_config

class ConfigGenerator:
    def __init__(self):
        self.console = console
        self.project_root = Path(__file__).parent.parent
        self.presets_dir = self.project_root / "presets"
        
        # Ensure presets directory exists
        self.presets_dir.mkdir(exist_ok=True)
    
    def generate_config(self, dataset_path=None, output_path=None):
        """Generate optimized configuration based on system and dataset"""
        
        self.console.print("\n[bold cyan]⚙️ Configuration Generator[/bold cyan]")
        self.console.print("=" * 50)
        
        # Get GPU info
        gpu_info = get_gpu_info()
        if not gpu_info:
            self.console.print("[red]❌ No CUDA GPU detected![/red]")
            return None
        
        # Display GPU info
        self._display_gpu_info(gpu_info)
        
        # Analyze dataset if provided
        dataset_info = None
        if dataset_path:
            dataset_info = self._analyze_dataset(dataset_path)
            if dataset_info:
                self._display_dataset_info(dataset_info)
        
        # Get recommended configuration
        preset = get_recommended_config(gpu_info['memory_gb'], dataset_info)
        
        # Generate full configuration
        config = self._create_full_config(preset, dataset_path, dataset_info)
        
        # Display configuration
        self._display_config(config, preset['name'])
        
        # Save configuration
        if output_path:
            config_path = output_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_name = f"auto_config_{preset['name'].lower().replace(' ', '_')}_{timestamp}.yml"
            config_path = self.presets_dir / config_name
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        self.console.print(f"\n[green]✅ Configuration saved to: {config_path}[/green]")
        
        return config_path
    
    def _display_gpu_info(self, gpu_info):
        """Display GPU information"""
        table = Table(title="🖥️ GPU Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("GPU Name", gpu_info['name'])
        table.add_row("Total Memory", f"{gpu_info['memory_gb']:.1f} GB")
        
        if 'memory_free_gb' in gpu_info:
            table.add_row("Free Memory", f"{gpu_info['memory_free_gb']:.1f} GB")
            table.add_row("Used Memory", f"{gpu_info['memory_used_gb']:.1f} GB")
        
        compute = gpu_info['compute_capability']
        table.add_row("Compute Capability", f"{compute[0]}.{compute[1]}")
        
        self.console.print(table)
    
    def _analyze_dataset(self, dataset_path):
        """Analyze dataset to optimize configuration"""
        try:
            # Find pickle files
            pickle_files = []
            for ext in ['*.pkl', '*.pickle']:
                pickle_files.extend(Path(dataset_path).glob(f"**/{ext}"))
            
            if not pickle_files:
                return None
            
            total_sequences = 0
            sequence_lengths = []
            
            # Sample first few files
            for file_path in pickle_files[:5]:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    
                    if isinstance(data, list) and data:
                        if isinstance(data[0], (list, tuple)):
                            total_sequences += len(data)
                            # Sample sequence lengths
                            for seq in data[:100]:
                                sequence_lengths.append(len(seq))
            
            if sequence_lengths:
                avg_length = sum(sequence_lengths) / len(sequence_lengths)
                max_length = max(sequence_lengths)
                min_length = min(sequence_lengths)
                
                return {
                    'num_files': len(pickle_files),
                    'sampled_sequences': total_sequences,
                    'avg_sequence_length': avg_length,
                    'max_sequence_length': max_length,
                    'min_sequence_length': min_length
                }
            
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not analyze dataset: {e}[/yellow]")
        
        return None
    
    def _display_dataset_info(self, dataset_info):
        """Display dataset information"""
        table = Table(title="📊 Dataset Analysis")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Dataset Files", str(dataset_info['num_files']))
        table.add_row("Sampled Sequences", str(dataset_info['sampled_sequences']))
        table.add_row("Avg Sequence Length", f"{dataset_info['avg_sequence_length']:.0f} tokens")
        table.add_row("Length Range", f"{dataset_info['min_sequence_length']} - {dataset_info['max_sequence_length']} tokens")
        
        self.console.print(table)
    
    def _create_full_config(self, preset, dataset_path, dataset_info):
        """Create full configuration from preset"""
        
        # Adjust sequence length based on dataset if available
        seq_len = preset['seq_len']
        if dataset_info and 'avg_sequence_length' in dataset_info:
            # Ensure seq_len is appropriate for the dataset
            avg_len = dataset_info['avg_sequence_length']
            if avg_len * 1.5 < seq_len:
                # Dataset has shorter sequences, we can use smaller seq_len
                seq_len = min(seq_len, int(avg_len * 2))
                seq_len = max(256, seq_len)  # Minimum 256
                # Round to nearest power of 2
                seq_len = 2 ** round(__import__('math').log2(seq_len))
        
        config = {
            '# Auto-generated configuration': f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            '# GPU': preset['name'],
            
            'dataset': {
                'path': str(dataset_path) if dataset_path else './DATA'
            },
            
            'model': {
                'seq_len': seq_len,
                'dim': preset['dim'],
                'depth': preset['depth'],
                'heads': preset['heads']
            },
            
            'training': {
                'batch_size': preset['batch_size'],
                'gradient_accumulate_every': preset['gradient_accumulate_every'],
                'learning_rate': 0.0001,
                'num_epochs': 5,
                'validate_every': 100,
                'save_every': 500,
                'generate_every': 250,
                'print_stats_every': 10,
                'grad_clip': 1.0,
                'generate_length': 512
            },
            
            'output': {
                'model_dir': './models',
                'sample_dir': './samples'
            }
        }
        
        return config
    
    def _display_config(self, config, preset_name):
        """Display generated configuration"""
        panel_content = f"""[bold green]Recommended Configuration: {preset_name}[/bold green]

[cyan]Model Architecture:[/cyan]
  • Sequence Length: {config['model']['seq_len']} tokens
  • Model Dimension: {config['model']['dim']}
  • Transformer Layers: {config['model']['depth']}
  • Attention Heads: {config['model']['heads']}

[cyan]Training Parameters:[/cyan]
  • Batch Size: {config['training']['batch_size']}
  • Gradient Accumulation: {config['training']['gradient_accumulate_every']}
  • Effective Batch Size: {config['training']['batch_size'] * config['training']['gradient_accumulate_every']}
  • Learning Rate: {config['training']['learning_rate']}
  • Epochs: {config['training']['num_epochs']}

[cyan]Memory Usage Estimate:[/cyan]
  • Model Parameters: ~{self._estimate_model_size(config['model'])} million
  • Estimated VRAM Usage: ~{self._estimate_vram_usage(config)} GB
"""
        
        panel = Panel(
            panel_content,
            title="[bold]Generated Configuration[/bold]",
            border_style="green"
        )
        
        self.console.print(panel)
    
    def _estimate_model_size(self, model_config):
        """Estimate model parameter count in millions"""
        # Rough estimation based on transformer architecture
        vocab_size = 18820  # Fixed vocabulary
        dim = model_config['dim']
        depth = model_config['depth']
        heads = model_config['heads']
        
        # Embedding + position encoding
        params = vocab_size * dim + model_config['seq_len'] * dim
        
        # Transformer layers (very rough estimate)
        params += depth * (
            4 * dim * dim +  # FFN
            4 * dim * dim    # Attention
        )
        
        return round(params / 1_000_000, 1)
    
    def _estimate_vram_usage(self, config):
        """Estimate VRAM usage in GB"""
        # Very rough estimation
        model_size_mb = self._estimate_model_size(config['model'])
        
        # Model + optimizer states + gradients + activations
        total_mb = model_size_mb * 4  # Rough multiplier for training
        
        # Add batch size factor
        batch_factor = config['training']['batch_size'] * config['model']['seq_len'] / 1024
        total_mb += batch_factor * 100  # Rough estimate for activations
        
        return round(total_mb / 1024, 1) 