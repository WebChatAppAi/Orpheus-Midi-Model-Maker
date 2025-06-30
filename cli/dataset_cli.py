"""
Dataset CLI Module - Handles dataset creation and validation
"""

import os
import sys
import glob
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn, MofNCompleteColumn
from rich import box
from rich.align import Align
from rich.columns import Columns
import time

# Add paths for existing modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "DatasetCreation"))

from .utils import (
    console, validate_path, print_section_header, create_status_table,
    show_spinner, show_success, show_error, show_warning, show_info,
    confirm_action, create_progress_panel
)

class DatasetCLI:
    def __init__(self):
        self.console = console
        self.project_root = project_root
        self.data_dir = self.project_root / "DATA"
        
        # Ensure DATA directory exists
        self.data_dir.mkdir(exist_ok=True)
    
    def create_dataset(self):
        """Handle dataset creation from MIDI files"""
        print_section_header("Create Dataset from MIDI Files", style="bold cyan")
        
        # Show instructions
        instructions = Panel(
            "[bold white]This wizard will guide you through creating a training dataset from MIDI files.[/bold white]\n\n"
            "[dim]Requirements:[/dim]\n"
            "• Directory containing MIDI files (.mid, .midi)\n"
            "• Files can be in subdirectories\n"
            "• Minimum 8 notes per file\n\n"
            "[yellow]Tip: More diverse MIDI files = better model![/yellow]",
            title="[bold cyan]Dataset Creation Wizard[/bold cyan]",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        self.console.print(instructions)
        self.console.print()
        
        # Get MIDI directory path
        while True:
            midi_path = Prompt.ask(
                "[bold cyan]Enter path to MIDI files directory[/bold cyan]",
                default="."
            )
            
            path = validate_path(midi_path, must_exist=True)
            if path and path.is_dir():
                break
            else:
                show_error(f"Directory not found: {midi_path}")
                if not confirm_action("Try again?"):
                    return False
        
        # Scan for MIDI files with progress
        self.console.print()
        with console.status("[cyan]Scanning for MIDI files...[/cyan]", spinner="dots"):
            midi_files = self._scan_midi_files(path)
        
        if not midi_files:
            show_error("No MIDI files found in the specified directory!")
            return False
        
        # Show summary with beautiful display
        self._display_midi_summary(midi_files, path)
        
        # Confirm processing
        if not confirm_action(f"Process {len(midi_files)} MIDI files?"):
            show_warning("Dataset creation cancelled")
            return False
        
        # Process MIDI files
        self.console.print()
        show_info("Starting MIDI processing...")
        
        from processdata import process_midi_dataset
        process_midi_dataset(midi_files)
        
        # Show post-processing options
        return self._show_post_processing_menu()
    
    def validate_dataset(self):
        """Validate existing dataset"""
        print_section_header("Validate Dataset", style="bold cyan")
        
        # Check for processed dataset
        processed_path = self.data_dir / "processed_dataset"
        melody_path = self.data_dir / "melody_dataset.pkl"
        
        suggestions = []
        if melody_path.exists():
            suggestions.append(str(melody_path))
        if processed_path.exists():
            suggestions.append(str(processed_path))
        
        if suggestions:
            show_info(f"Found existing dataset at: {suggestions[0]}")
        
        # Get dataset path
        dataset_path = Prompt.ask(
            "[bold cyan]Enter path to dataset directory or file[/bold cyan]",
            default=suggestions[0] if suggestions else str(self.data_dir)
        )
        
        path = validate_path(dataset_path, must_exist=True)
        if not path:
            show_error(f"Path not found: {dataset_path}")
            return False
        
        # Run validation with progress
        self.console.print()
        with console.status("[cyan]Validating dataset...[/cyan]", spinner="dots"):
            time.sleep(0.5)  # Brief pause for visual effect
        
        from verifydata import verify_dataset
        result = verify_dataset(str(path))
        
        if result:
            show_success("Dataset validation PASSED! ✨")
            
            # Show option to proceed to training
            if confirm_action("Would you like to proceed to training?"):
                return "training"
        else:
            show_error("Dataset validation FAILED!")
            show_info("Please check the errors above and fix your dataset.")
        
        return result
    
    def _scan_midi_files(self, directory):
        """Scan directory for MIDI files with better progress display"""
        midi_extensions = ['*.mid', '*.midi', '*.MID', '*.MIDI']
        midi_files = []
        
        # First, count total files for progress
        total_files = 0
        for ext in midi_extensions:
            pattern = os.path.join(directory, '**', ext)
            total_files += len(glob.glob(pattern, recursive=True))
        
        if total_files == 0:
            return []
        
        # Now scan with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("[cyan]Scanning MIDI files...[/cyan]", total=total_files)
            
            for ext in midi_extensions:
                pattern = os.path.join(directory, '**', ext)
                files = glob.glob(pattern, recursive=True)
                for file in files:
                    if file not in midi_files:
                        midi_files.append(file)
                        progress.update(task, advance=1)
        
        return sorted(midi_files)
    
    def _display_midi_summary(self, midi_files, base_path):
        """Display summary of found MIDI files with beautiful formatting"""
        # Directory breakdown
        directories = {}
        total_size = 0
        
        for file in midi_files:
            dir_name = os.path.dirname(file)
            rel_dir = os.path.relpath(dir_name, base_path)
            if rel_dir == '.':
                rel_dir = 'Root directory'
            directories[rel_dir] = directories.get(rel_dir, 0) + 1
            
            # Get file size
            try:
                total_size += os.path.getsize(file)
            except:
                pass
        
        # Create summary panels
        summary_items = [
            ("Total MIDI Files", f"{len(midi_files):,}"),
            ("Total Size", f"{total_size / (1024*1024):.1f} MB"),
            ("Directories", str(len(directories))),
            ("Average Files/Directory", f"{len(midi_files) / max(len(directories), 1):.1f}")
        ]
        
        summary_table = create_status_table("Dataset Summary", summary_items, style="green")
        
        # Create directory breakdown table
        dir_table = Table(
            title="[bold cyan]Directory Breakdown[/bold cyan]",
            box=box.ROUNDED,
            show_lines=True
        )
        dir_table.add_column("Directory", style="cyan", ratio=2)
        dir_table.add_column("Files", justify="right", style="green", ratio=1)
        dir_table.add_column("Percentage", justify="right", style="yellow", ratio=1)
        
        # Sort directories by file count
        sorted_dirs = sorted(directories.items(), key=lambda x: x[1], reverse=True)
        
        for dir_name, count in sorted_dirs[:10]:  # Show top 10
            percentage = (count / len(midi_files)) * 100
            dir_table.add_row(
                dir_name[:50] + "..." if len(dir_name) > 50 else dir_name,
                str(count),
                f"{percentage:.1f}%"
            )
        
        if len(sorted_dirs) > 10:
            dir_table.add_row(
                f"[dim]... and {len(sorted_dirs) - 10} more directories[/dim]",
                "[dim]...[/dim]",
                "[dim]...[/dim]"
            )
        
        # Display everything
        self.console.print()
        self.console.print(Columns([summary_table, dir_table], padding=2, expand=False))
        
        # Show sample files
        self.console.print()
        sample_panel = Panel(
            "\n".join([f"• {os.path.basename(f)}" for f in midi_files[:5]]) +
            (f"\n[dim]... and {len(midi_files) - 5} more files[/dim]" if len(midi_files) > 5 else ""),
            title="[bold cyan]Sample Files[/bold cyan]",
            box=box.ROUNDED
        )
        self.console.print(sample_panel)
    
    def _show_post_processing_menu(self):
        """Show options after dataset creation with beautiful UI"""
        self.console.print()
        
        success_panel = Panel(
            Align.center(
                "[bold green]✅ Dataset created successfully![/bold green]\n\n"
                "[white]Your MIDI files have been processed and saved.[/white]\n"
                "[dim]Dataset location: ./DATA/melody_dataset.pkl[/dim]",
                vertical="middle"
            ),
            box=box.DOUBLE,
            style="green",
            padding=(1, 2),
            title="[bold]Success[/bold]"
        )
        self.console.print(success_panel)
        
        # Options table
        options = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
        options.add_column(style="cyan", width=5, justify="center")
        options.add_column(style="white")
        
        options.add_row("1", "Create another dataset from different MIDI files")
        options.add_row("2", "Validate the processed dataset")
        options.add_row("3", "Proceed to training setup")
        options.add_row("4", "Return to main menu")
        
        self.console.print()
        self.console.print(Align.center(options))
        
        choice = Prompt.ask(
            "\n[bold cyan]What would you like to do next?[/bold cyan]",
            choices=["1", "2", "3", "4"],
            default="3"
        )
        
        if choice == "1":
            return self.create_dataset()
        elif choice == "2":
            return self.validate_dataset()
        elif choice == "3":
            return "training"
        
        return None 