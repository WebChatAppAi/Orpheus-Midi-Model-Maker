"""
Main Menu Module - Central interface for Orpheus Midi Model Maker
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich import box
from rich.align import Align
from rich.text import Text
import time

from .utils import (
    console, display_banner, show_spinner, confirm_action,
    show_success, show_error, show_warning, print_section_header,
    create_status_table
)
from .dataset_cli import DatasetCLI
from .training_cli import TrainingCLI

class MainMenu:
    def __init__(self):
        self.console = console
        self.dataset_cli = DatasetCLI()
        self.training_cli = TrainingCLI()
        self.running = True
    
    def run(self):
        """Main application loop"""
        # Clear screen
        self.console.clear()
        
        # Display banner
        display_banner()
        
        # Check environment with spinner
        with console.status("[cyan]Checking environment...[/cyan]", spinner="dots"):
            env_ok = self._check_environment()
        
        if not env_ok:
            show_error("Please fix environment issues before continuing.")
            sys.exit(1)
        
        show_success("Environment check passed!")
        time.sleep(1)
        
        # Show quick stats
        self._show_quick_stats()
        
        # Main menu loop
        while self.running:
            choice = self._show_menu()
            self._handle_choice(choice)
        
        # Exit message
        self._show_exit_message()
    
    def _check_environment(self):
        """Quick environment check"""
        try:
            # Check if tegridy-tools exists
            project_root = Path(__file__).parent.parent
            tegridy_path = project_root / "tegridy-tools" / "tegridy-tools"
            
            if not tegridy_path.exists():
                return False
            
            # Check for DATA directory
            data_dir = project_root / "DATA"
            data_dir.mkdir(exist_ok=True)
            
            # Try importing key modules
            sys.path.append(str(tegridy_path))
            try:
                import TMIDIX
            except ImportError:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _show_menu(self):
        """Display main menu and get user choice"""
        print_section_header("Main Menu")
        
        # Create menu table
        menu_table = Table(
            box=box.DOUBLE_EDGE,
            title="[bold cyan]Choose an Option[/bold cyan]",
            title_style="bold cyan",
            width=80,
            padding=(1, 3)
        )
        
        menu_table.add_column("Option", style="bold cyan", width=10, justify="center")
        menu_table.add_column("Feature", style="bold white", width=25)
        menu_table.add_column("Description", style="dim white", width=35)
        
        menu_table.add_row(
            "1",
            "🎼 Create Dataset",
            "Process MIDI files into training data"
        )
        menu_table.add_row(
            "2",
            "✅ Validate Dataset",
            "Check existing dataset integrity"
        )
        menu_table.add_row(
            "3",
            "🚀 Start Training",
            "Train your AI melody model"
        )
        menu_table.add_section()
        menu_table.add_row(
            "4",
            "❌ Exit",
            "Close the application"
        )
        
        self.console.print(Align.center(menu_table))
        
        # Get user choice with styled prompt
        while True:
            choice = Prompt.ask(
                "\n[bold cyan]Select option[/bold cyan]",
                choices=["1", "2", "3", "4"],
                default="1"
            )
            return choice
    
    def _handle_choice(self, choice):
        """Handle menu choice"""
        if choice == "1":
            # Create Dataset
            self.console.clear()
            display_banner()
            result = self.dataset_cli.create_dataset()
            
            # Check if user wants to proceed to training
            if result == "training":
                self._handle_choice("3")
            else:
                self._pause_and_continue()
                
        elif choice == "2":
            # Validate Dataset
            self.console.clear()
            display_banner()
            self.dataset_cli.validate_dataset()
            self._pause_and_continue()
            
        elif choice == "3":
            # Start Training
            self.console.clear()
            display_banner()
            self.training_cli.start_training()
            self._pause_and_continue()
            
        elif choice == "4":
            # Exit
            if confirm_action("Are you sure you want to exit?"):
                self.running = False
    
    def _pause_and_continue(self):
        """Pause and wait for user to continue"""
        console.input("\n[dim]Press Enter to return to main menu...[/dim]")
        self.console.clear()
        display_banner()
        self._show_quick_stats()
    
    def _show_quick_stats(self):
        """Show quick statistics"""
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "DATA"
        models_dir = project_root / "models"
        samples_dir = project_root / "samples"
        
        stats = []
        
        # Check for datasets
        if data_dir.exists():
            pkl_files = list(data_dir.glob("*.pkl"))
            if pkl_files:
                stats.append(("Datasets", f"{len(pkl_files)} files found"))
            else:
                stats.append(("Datasets", "No datasets found"))
        
        # Check for models
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pth"))
            if model_files:
                stats.append(("Saved Models", f"{len(model_files)} models"))
        
        # Check for samples
        if samples_dir.exists():
            sample_files = list(samples_dir.glob("*.mid"))
            if sample_files:
                stats.append(("Generated Samples", f"{len(sample_files)} MIDI files"))
        
        if stats:
            stats_table = create_status_table("Project Status", stats, style="green")
            self.console.print(Align.center(stats_table))
            self.console.print()
    
    def _show_exit_message(self):
        """Show exit message with style"""
        exit_panel = Panel(
            Align.center(
                "[bold green]Thank you for using Orpheus Midi Model Maker![/bold green]\n\n"
                "[dim]Happy music making! 🎵[/dim]",
                vertical="middle"
            ),
            box=box.DOUBLE,
            style="green",
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(exit_panel)
        self.console.print() 