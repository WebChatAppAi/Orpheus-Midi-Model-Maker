"""
Training CLI Module - Handles training setup and execution
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.align import Align
from rich.columns import Columns
import torch

# Add paths for existing modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "TrainingModel"))

from .utils import (
    console, validate_path, get_gpu_info, create_status_table,
    print_section_header, show_success, show_error, show_warning,
    show_info, confirm_action, show_spinner, format_time
)
from .config_generator import ConfigGenerator
from .dashboard import TrainingDashboard

class TrainingCLI:
    def __init__(self):
        self.console = console
        self.project_root = project_root
        self.config_generator = ConfigGenerator()
        self.data_dir = self.project_root / "DATA"
    
    def start_training(self):
        """Main training setup and launch"""
        print_section_header("Start Training", style="bold cyan")
        
        # Check GPU with fancy display
        self._check_gpu_requirements()
        
        # Get or generate configuration
        config_path = self._get_configuration()
        if not config_path:
            return False
        
        # Load and validate configuration
        config = self._load_and_validate_config(config_path)
        if not config:
            return False
        
        # Display training summary
        self._display_training_summary(config, config_path)
        
        # Confirm before starting
        if not confirm_action("Ready to start training?"):
            show_warning("Training cancelled")
            return False
        
        # Launch training
        self._launch_training(config_path)
        
        return True
    
    def _check_gpu_requirements(self):
        """Check GPU availability with nice display"""
        with console.status("[cyan]Checking GPU availability...[/cyan]", spinner="dots"):
            gpu_info = get_gpu_info()
        
        if not gpu_info:
            show_error("No CUDA GPU detected! Training requires a NVIDIA GPU.")
            self.console.print("\n[yellow]Please ensure:[/yellow]")
            self.console.print("  • NVIDIA GPU is installed")
            self.console.print("  • CUDA drivers are installed")
            self.console.print("  • PyTorch is installed with CUDA support")
            sys.exit(1)
        
        # Display GPU info
        gpu_items = [
            ("GPU Model", gpu_info['name']),
            ("Total Memory", f"{gpu_info['memory_gb']:.1f} GB"),
        ]
        
        if 'memory_free_gb' in gpu_info:
            gpu_items.extend([
                ("Free Memory", f"{gpu_info['memory_free_gb']:.1f} GB"),
                ("Used Memory", f"{gpu_info['memory_used_gb']:.1f} GB")
            ])
        
        gpu_table = create_status_table("GPU Information", gpu_items, style="green")
        self.console.print(gpu_table)
        self.console.print()
    
    def _get_configuration(self):
        """Get configuration file path or generate one"""
        # Check for existing configs
        existing_configs = self._find_existing_configs()
        
        # Create options table
        options = Table(
            box=box.DOUBLE_EDGE,
            title="[bold cyan]Configuration Options[/bold cyan]",
            show_header=False,
            padding=(1, 2)
        )
        options.add_column(style="cyan", width=5, justify="center")
        options.add_column(style="white")
        
        if existing_configs:
            options.add_row("1", "📄 Use existing configuration file")
        
        options.add_row("2", "🤖 Auto-generate optimal configuration")
        options.add_row("3", "📝 Provide custom configuration path")
        
        self.console.print(options)
        
        choices = ["2", "3"]
        if existing_configs:
            choices.insert(0, "1")
        
        choice = Prompt.ask(
            "\n[bold cyan]Select option[/bold cyan]",
            choices=choices,
            default="2" if not existing_configs else "1"
        )
        
        if choice == "1" and existing_configs:
            return self._select_existing_config(existing_configs)
        elif choice == "2":
            return self._auto_generate_config()
        elif choice == "3":
            return self._get_custom_config_path()
    
    def _auto_generate_config(self):
        """Auto-generate configuration with nice flow"""
        show_info("Auto-generating optimal configuration based on your system...")
        
        # Get dataset path
        dataset_path = self._get_dataset_path()
        if not dataset_path:
            return None
        
        # Generate config
        self.console.print()
        with console.status("[cyan]Analyzing system and dataset...[/cyan]", spinner="dots"):
            config_path = self.config_generator.generate_config(dataset_path)
        
        if config_path:
            show_success(f"Configuration generated: {config_path}")
            
            # Ask if user wants to edit
            if confirm_action("Would you like to edit the configuration?", default=False):
                self._edit_config_interactive(config_path)
        
        return config_path
    
    def _find_existing_configs(self):
        """Find existing configuration files"""
        configs = []
        
        # Check presets directory
        presets_dir = self.project_root / "presets"
        if presets_dir.exists():
            configs.extend(presets_dir.glob("*.yml"))
            configs.extend(presets_dir.glob("*.yaml"))
        
        # Check TrainingModel directory
        training_dir = self.project_root / "TrainingModel"
        if training_dir.exists():
            for pattern in ["*.yml", "*.yaml"]:
                configs.extend(training_dir.glob(pattern))
        
        return sorted(configs, key=lambda x: x.stat().st_mtime, reverse=True)[:10]
    
    def _select_existing_config(self, configs):
        """Select from existing configuration files"""
        table = Table(
            title="[bold cyan]Available Configurations[/bold cyan]",
            box=box.ROUNDED,
            show_lines=True
        )
        table.add_column("#", style="cyan", width=5, justify="center")
        table.add_column("File", style="white", width=40)
        table.add_column("Modified", style="dim", width=20)
        table.add_column("Size", style="dim", width=10, justify="right")
        
        for i, config in enumerate(configs, 1):
            modified = config.stat().st_mtime
            from datetime import datetime
            mod_time = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
            size = config.stat().st_size
            
            table.add_row(
                str(i),
                config.name,
                mod_time,
                f"{size / 1024:.1f} KB"
            )
        
        self.console.print()
        self.console.print(table)
        
        choice = Prompt.ask(
            "\n[bold cyan]Select configuration[/bold cyan]",
            choices=[str(i) for i in range(1, len(configs) + 1)]
        )
        
        return configs[int(choice) - 1]
    
    def _get_custom_config_path(self):
        """Get custom configuration path"""
        while True:
            config_path = Prompt.ask("[bold cyan]Enter path to configuration file (.yml)[/bold cyan]")
            path = validate_path(config_path, must_exist=True)
            
            if path and path.suffix in ['.yml', '.yaml']:
                return path
            else:
                show_error(f"Invalid configuration file: {config_path}")
                if not confirm_action("Try again?"):
                    return None
    
    def _get_dataset_path(self):
        """Get dataset path for configuration generation"""
        # Check for default dataset
        default_datasets = [
            self.data_dir / "melody_dataset.pkl",
            self.data_dir / "processed_dataset"
        ]
        
        found_dataset = None
        for dataset in default_datasets:
            if dataset.exists():
                found_dataset = dataset
                break
        
        if found_dataset:
            show_info(f"Found dataset at: {found_dataset}")
            if confirm_action("Use this dataset?"):
                return found_dataset.parent if found_dataset.is_file() else found_dataset
        
        # Ask for custom path
        dataset_path = Prompt.ask(
            "[bold cyan]Enter path to dataset directory[/bold cyan]",
            default=str(self.data_dir)
        )
        
        path = validate_path(dataset_path, must_exist=True)
        if not path:
            show_error(f"Directory not found: {dataset_path}")
            return None
        
        return path
    
    def _load_and_validate_config(self, config_path):
        """Load and validate configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Basic validation
            required_sections = ['dataset', 'model', 'training', 'output']
            missing = [s for s in required_sections if s not in config]
            
            if missing:
                show_error(f"Invalid configuration: missing sections {missing}")
                return None
            
            # Validate dataset path
            dataset_path = Path(config['dataset']['path'])
            if not dataset_path.exists():
                show_error(f"Dataset path not found: {dataset_path}")
                return None
            
            return config
            
        except Exception as e:
            show_error(f"Error loading configuration: {e}")
            return None
    
    def _display_training_summary(self, config, config_path):
        """Display training summary before starting"""
        # Get GPU info again for display
        gpu_info = get_gpu_info()
        
        # Create configuration table
        config_table = Table(
            title="[bold cyan]Training Configuration[/bold cyan]",
            box=box.DOUBLE_EDGE,
            show_lines=True,
            title_style="bold cyan"
        )
        config_table.add_column("Category", style="cyan", width=20)
        config_table.add_column("Parameter", style="white", width=25)
        config_table.add_column("Value", style="green", width=30)
        
        # Configuration file
        config_table.add_row(
            "Configuration",
            "File",
            str(Path(config_path).name)
        )
        config_table.add_section()
        
        # Dataset
        config_table.add_row("Dataset", "Path", str(config['dataset']['path']))
        config_table.add_section()
        
        # Model architecture
        config_table.add_row("Model", "Sequence Length", f"{config['model']['seq_len']:,} tokens")
        config_table.add_row("", "Model Dimension", str(config['model']['dim']))
        config_table.add_row("", "Transformer Layers", str(config['model']['depth']))
        config_table.add_row("", "Attention Heads", str(config['model']['heads']))
        
        # Calculate model size
        model_params = self._estimate_model_parameters(config['model'])
        config_table.add_row("", "Parameters", f"~{model_params}M")
        config_table.add_section()
        
        # Training parameters
        config_table.add_row("Training", "Batch Size", str(config['training']['batch_size']))
        config_table.add_row("", "Gradient Accumulation", str(config['training']['gradient_accumulate_every']))
        
        effective_batch = config['training']['batch_size'] * config['training']['gradient_accumulate_every']
        config_table.add_row("", "Effective Batch Size", str(effective_batch))
        config_table.add_row("", "Learning Rate", f"{config['training']['learning_rate']:.2e}")
        config_table.add_row("", "Epochs", str(config['training']['num_epochs']))
        config_table.add_section()
        
        # Output paths
        config_table.add_row("Output", "Model Directory", config['output']['model_dir'])
        config_table.add_row("", "Sample Directory", config['output']['sample_dir'])
        
        self.console.print()
        self.console.print(config_table)
        
        # Show estimated training time
        self._show_training_estimates(config, gpu_info)
    
    def _estimate_model_parameters(self, model_config):
        """Estimate model parameters in millions"""
        vocab_size = 18820  # Fixed vocabulary
        dim = model_config['dim']
        depth = model_config['depth']
        
        # Rough estimation
        params = vocab_size * dim  # Embeddings
        params += depth * (4 * dim * dim + 4 * dim * dim)  # Transformer layers
        
        return round(params / 1_000_000, 1)
    
    def _show_training_estimates(self, config, gpu_info):
        """Show training time estimates"""
        # Rough estimates based on GPU and config
        gpu_memory = gpu_info['memory_gb'] if gpu_info else 8
        batch_size = config['training']['batch_size']
        epochs = config['training']['num_epochs']
        
        # Very rough speed estimates (steps per second)
        if gpu_memory >= 24:
            speed = 5.0
        elif gpu_memory >= 16:
            speed = 3.0
        elif gpu_memory >= 12:
            speed = 2.0
        else:
            speed = 1.0
        
        # Adjust for batch size
        speed = speed * (batch_size / 2)
        
        # Estimate total time (assuming 1000 steps per epoch)
        total_steps = epochs * 1000
        total_seconds = total_steps / speed
        
        estimates = [
            ("Estimated Speed", f"~{speed:.1f} steps/sec"),
            ("Estimated Time/Epoch", format_time(1000 / speed)),
            ("Total Training Time", format_time(total_seconds)),
        ]
        
        estimate_table = create_status_table("Training Estimates", estimates, style="yellow")
        
        self.console.print()
        self.console.print(estimate_table)
    
    def _launch_training(self, config_path):
        """Launch the training process"""
        print_section_header("Launching Training", style="bold green")
        
        # Ask about dashboard
        use_dashboard = confirm_action("Use live training dashboard?")
        
        if use_dashboard:
            # Use custom dashboard
            show_info("Starting training with live dashboard...")
            dashboard = TrainingDashboard()
            dashboard.run_training(config_path)
        else:
            # Launch original training script
            show_info("Starting training in standard mode...")
            training_script = self.project_root / "TrainingModel" / "app.py"
            
            try:
                # Set config path in environment
                env = os.environ.copy()
                env['CONFIG_PATH'] = str(config_path)
                
                # Run training
                subprocess.run([
                    sys.executable,
                    str(training_script)
                ], env=env)
                
                show_success("Training completed!")
                
            except KeyboardInterrupt:
                show_warning("Training interrupted by user")
            except Exception as e:
                show_error(f"Training failed: {e}")
    
    def _edit_config_interactive(self, config_path):
        """Interactive config editing"""
        import platform
        
        show_info(f"Opening configuration file: {config_path}")
        
        try:
            if platform.system() == 'Windows':
                os.startfile(config_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(['open', config_path])
            else:  # Linux
                subprocess.call(['xdg-open', config_path])
            
            self.console.print("\n[dim]Edit the configuration file in your editor.[/dim]")
            console.input("[dim]Press Enter when done editing...[/dim]")
            
        except Exception as e:
            show_warning(f"Could not open editor: {e}")
            self.console.print("Please edit the file manually:", config_path) 