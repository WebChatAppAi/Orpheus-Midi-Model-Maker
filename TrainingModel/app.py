"""
MIDI Model Creator - Advanced Training Module
Main entry point with beautiful CLI interface
"""

import os
import sys
import yaml
import torch
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

# Add project paths
project_root = Path(__file__).parent.parent
tegridy_path = project_root / "tegridy-tools" / "tegridy-tools"
xtransformer_path = tegridy_path / "X-Transformer"
sys.path.append(str(tegridy_path))
sys.path.append(str(xtransformer_path))

from config import ConfigManager
from trainer import MelodyTrainer
from utils import CLIUtils

console = Console()

class TrainingApp:
    def __init__(self):
        self.console = console
        self.cli_utils = CLIUtils()
        self.config_manager = ConfigManager()
        
    def show_banner(self):
        """Display application banner"""
        banner = Text()
        banner.append("🎵 MIDI MODEL CREATOR 🎵\n", style="bold magenta")
        banner.append("Advanced Melody Transformer Training\n", style="bold cyan")
        banner.append("Powered by tegridy-tools & X-Transformer", style="dim")
        
        panel = Panel(
            banner,
            title="[bold green]Training Module v1.0[/bold green]",
            border_style="bright_blue",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def check_environment(self):
        """Comprehensive environment checking"""
        self.console.print("\n🔍 [bold yellow]Environment Check[/bold yellow]")
        self.console.print("=" * 50)
        
        checks = []
        
        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 8):
            checks.append(("Python Version", f"{python_version}", "✅"))
        else:
            checks.append(("Python Version", f"{python_version} (Need 3.8+)", "❌"))
            
        # Check CUDA availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            checks.append(("CUDA", f"Available - {gpu_name}", "✅"))
            checks.append(("GPU Memory", f"{gpu_memory:.1f} GB", "✅"))
        else:
            checks.append(("CUDA", "Not Available", "❌"))
            
        # Check tegridy-tools
        try:
            import TMIDIX
            checks.append(("TMIDIX", "Available", "✅"))
        except ImportError:
            checks.append(("TMIDIX", "Not Found", "❌"))
            
        # Check X-Transformer
        try:
            from x_transformer_2_3_1 import TransformerWrapper, Decoder, AutoregressiveWrapper
            checks.append(("X-Transformer", "Available", "✅"))
        except ImportError:
            checks.append(("X-Transformer", "Not Found", "❌"))
            
        # Check required packages
        required_packages = ['torch', 'tqdm', 'matplotlib', 'sklearn', 'yaml', 'rich']
        for package in required_packages:
            try:
                __import__(package)
                checks.append((f"{package}", "Installed", "✅"))
            except ImportError:
                checks.append((f"{package}", "Missing", "❌"))
        
        # Display results
        table = Table(title="Environment Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Result", justify="center")
        
        for component, status, result in checks:
            table.add_row(component, status, result)
            
        self.console.print(table)
        
        # Check if all critical components are available
        critical_failed = any(check[2] == "❌" and check[0] in ["CUDA", "TMIDIX", "X-Transformer"] for check in checks)
        
        if critical_failed:
            self.console.print("\n❌ [bold red]Critical components missing![/bold red]")
            self.console.print("💡 Please run setup.sh to install dependencies")
            return False
            
        self.console.print("\n✅ [bold green]Environment check passed![/bold green]")
        return True
    
    def get_config_path(self):
        """Get configuration file path from user"""
        self.console.print("\n📄 [bold yellow]Configuration Setup[/bold yellow]")
        self.console.print("=" * 30)
        
        # Show sample config location
        sample_config = project_root / "TrainingModel" / "sample_config.yml"
        self.console.print(f"💡 Sample config: {sample_config}")
        
        while True:
            config_path = Prompt.ask("Enter path to your training config (YML file)")
            
            if os.path.exists(config_path):
                if config_path.endswith(('.yml', '.yaml')):
                    return config_path
                else:
                    self.console.print("❌ File must be a YML/YAML file")
            else:
                self.console.print(f"❌ File not found: {config_path}")
                
                if Confirm.ask("Try again?"):
                    continue
                else:
                    return None
    
    def validate_config(self, config_path):
        """Validate configuration file"""
        self.console.print(f"\n🔍 [bold yellow]Validating Configuration[/bold yellow]")
        self.console.print("=" * 35)
        
        try:
            config = self.config_manager.load_config(config_path)
            validation_result = self.config_manager.validate_config(config)
            
            if validation_result['valid']:
                self.console.print("✅ [bold green]Configuration is valid![/bold green]")
                
                # Display config summary
                self.display_config_summary(config)
                return config
            else:
                self.console.print("❌ [bold red]Configuration validation failed:[/bold red]")
                for error in validation_result['errors']:
                    self.console.print(f"   • {error}")
                return None
                
        except Exception as e:
            self.console.print(f"❌ [bold red]Error loading config: {e}[/bold red]")
            return None
    
    def display_config_summary(self, config):
        """Display configuration summary"""
        table = Table(title="Training Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")
        
        # Dataset info
        table.add_row("Dataset Path", config['dataset']['path'])
        table.add_row("Sequence Length", str(config['model']['seq_len']))
        
        # Model parameters
        table.add_row("Model Dimension", str(config['model']['dim']))
        table.add_row("Layers", str(config['model']['depth']))
        table.add_row("Attention Heads", str(config['model']['heads']))
        
        # Training parameters
        table.add_row("Batch Size", str(config['training']['batch_size']))
        table.add_row("Gradient Accumulation", str(config['training']['gradient_accumulate_every']))
        table.add_row("Learning Rate", str(config['training']['learning_rate']))
        table.add_row("Epochs", str(config['training']['num_epochs']))
        
        # Output paths
        table.add_row("Model Output", config['output']['model_dir'])
        table.add_row("Sample Output", config['output']['sample_dir'])
        
        self.console.print(table)
    
    def start_training(self, config):
        """Start the training process"""
        self.console.print(f"\n🚀 [bold yellow]Starting Training Process[/bold yellow]")
        self.console.print("=" * 35)
        
        # Confirm before starting
        if not Confirm.ask("Ready to start training?", default=True):
            self.console.print("Training cancelled by user")
            return
        
        # Initialize trainer
        trainer = MelodyTrainer(config, self.console)
        
        try:
            # Start training
            trainer.train()
            
            self.console.print("\n🎉 [bold green]Training completed successfully![/bold green]")
            
        except KeyboardInterrupt:
            self.console.print("\n⚠️ [bold yellow]Training interrupted by user[/bold yellow]")
            trainer.save_checkpoint("interrupted")
            
        except Exception as e:
            self.console.print(f"\n❌ [bold red]Training failed: {e}[/bold red]")
            
    def main(self):
        """Main application loop"""
        self.show_banner()
        
        # Environment check
        if not self.check_environment():
            sys.exit(1)
        
        # Get configuration
        config_path = self.get_config_path()
        if not config_path:
            self.console.print("❌ No configuration provided. Exiting.")
            sys.exit(1)
        
        # Validate configuration
        config = self.validate_config(config_path)
        if not config:
            sys.exit(1)
        
        # Start training
        self.start_training(config)
        
        self.console.print("\n👋 Thanks for using MIDI Model Creator!")

if __name__ == "__main__":
    app = TrainingApp()
    app.main()