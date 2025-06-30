"""
Utility functions for the CLI interface
"""

import torch
import platform
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.align import Align
from rich.table import Table
import time

console = Console()

def get_gpu_info():
    """Get GPU information including memory"""
    if not torch.cuda.is_available():
        return None
    
    gpu_info = {
        'name': torch.cuda.get_device_name(0),
        'memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
        'compute_capability': torch.cuda.get_device_capability(0)
    }
    
    # Try to get current memory usage
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            used, total = map(int, result.stdout.strip().split(','))
            gpu_info['memory_used_gb'] = used / 1024
            gpu_info['memory_free_gb'] = (total - used) / 1024
    except:
        pass
    
    return gpu_info

def get_recommended_config(gpu_memory_gb, dataset_size=None):
    """Get recommended configuration based on GPU memory"""
    
    # Define presets based on GPU memory
    if gpu_memory_gb < 8:
        preset = {
            'name': 'Small GPU (< 8GB)',
            'seq_len': 512,
            'dim': 768,
            'depth': 4,
            'heads': 12,
            'batch_size': 1,
            'gradient_accumulate_every': 32
        }
    elif gpu_memory_gb < 12:
        preset = {
            'name': 'Medium GPU (8-12GB)',
            'seq_len': 1024,
            'dim': 1024,
            'depth': 6,
            'heads': 16,
            'batch_size': 2,
            'gradient_accumulate_every': 16
        }
    elif gpu_memory_gb < 16:
        preset = {
            'name': 'Medium-Large GPU (12-16GB)',
            'seq_len': 1536,
            'dim': 1280,
            'depth': 6,
            'heads': 20,
            'batch_size': 3,
            'gradient_accumulate_every': 12
        }
    elif gpu_memory_gb < 24:
        preset = {
            'name': 'Large GPU (16-24GB)',
            'seq_len': 2048,
            'dim': 1536,
            'depth': 8,
            'heads': 24,
            'batch_size': 4,
            'gradient_accumulate_every': 8
        }
    else:
        preset = {
            'name': 'Very Large GPU (24GB+)',
            'seq_len': 4096,
            'dim': 2048,
            'depth': 8,
            'heads': 32,
            'batch_size': 8,
            'gradient_accumulate_every': 4
        }
    
    return preset

def display_banner():
    """Display the main application banner with professional styling"""
    banner_text = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║              🎵 Orpheus Midi Model Maker 🎵                      ║
║                                                                   ║
║         Advanced AI Training & Dataset Creation Suite             ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """
    
    console.print(banner_text, style="bold cyan", justify="center")
    
    # Credits section
    credits = Table(box=None, show_header=False, padding=(0, 2))
    credits.add_column(justify="center")
    
    credits.add_row(
        "[dim white]Built on @asigalov61's tegridy-tools: [cyan]https://github.com/asigalov61/tegridy-tools[/cyan][/dim white]"
    )
    credits.add_row(
        "[dim white]CLI developed by: [cyan]https://github.com/WebChatAppAi[/cyan][/dim white]"
    )
    credits.add_row(
        "[dim white]After training, use: [cyan]https://github.com/WebChatAppAi/midi-gen[/cyan][/dim white]"
    )
    
    console.print(Align.center(credits))
    console.print()

def create_status_table(title, items, style="cyan"):
    """Create a nicely formatted status table"""
    table = Table(title=title, box=box.ROUNDED, title_style=f"bold {style}")
    table.add_column("Property", style="white", width=25)
    table.add_column("Value", style=style, width=35)
    
    for key, value in items:
        table.add_row(key, str(value))
    
    return table

def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"

def validate_path(path_str, must_exist=True, create_if_missing=False):
    """Validate and return a Path object"""
    path = Path(path_str).resolve()
    
    if must_exist and not path.exists():
        if create_if_missing and path.suffix == '':  # It's a directory
            path.mkdir(parents=True, exist_ok=True)
            return path
        else:
            return None
    
    return path

def show_spinner(message, duration=2):
    """Show a spinner with a message for a given duration"""
    with console.status(f"[cyan]{message}[/cyan]", spinner="dots"):
        time.sleep(duration)

def create_progress_panel(title, completed, total, show_percentage=True):
    """Create a progress panel with bar visualization"""
    progress = completed / total if total > 0 else 0
    bar_width = 40
    filled = int(bar_width * progress)
    bar = "█" * filled + "░" * (bar_width - filled)
    
    content = f"[green]{bar}[/green]"
    if show_percentage:
        content += f"\n\n[cyan]Progress:[/cyan] {progress*100:.1f}%"
        content += f"\n[cyan]Completed:[/cyan] {completed:,} / {total:,}"
    
    return Panel(
        Align.center(content, vertical="middle"),
        title=f"[bold]{title}[/bold]",
        box=box.ROUNDED,
        padding=(1, 2)
    )

def print_section_header(text, style="bold cyan"):
    """Print a section header with consistent styling"""
    console.print()
    console.rule(f"[{style}]{text}[/{style}]", style=style)
    console.print()

def confirm_action(message, default=True):
    """Show a confirmation prompt with consistent styling"""
    default_text = "[Y/n]" if default else "[y/N]"
    response = console.input(f"\n[bold yellow]{message} {default_text}:[/bold yellow] ").strip().lower()
    
    if not response:
        return default
    
    return response in ['y', 'yes']

def show_success(message):
    """Show a success message with consistent styling"""
    console.print(f"\n[bold green]✅ {message}[/bold green]\n")

def show_error(message):
    """Show an error message with consistent styling"""
    console.print(f"\n[bold red]❌ {message}[/bold red]\n")

def show_warning(message):
    """Show a warning message with consistent styling"""
    console.print(f"\n[bold yellow]⚠️  {message}[/bold yellow]\n")

def show_info(message):
    """Show an info message with consistent styling"""
    console.print(f"\n[bold blue]ℹ️  {message}[/bold blue]\n") 