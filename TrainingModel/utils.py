
"""
CLI Utilities for beautiful terminal interface
"""

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

class CLIUtils:
    def __init__(self):
        self.console = Console()
    
    def create_status_table(self, title, data):
        """Create a beautiful status table"""
        table = Table(title=title)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in data.items():
            table.add_row(str(key), str(value))
        
        return table
    
    def create_progress_panel(self, epoch, step, total_steps, loss, acc):
        """Create a training progress panel"""
        content = f"""
Epoch: {epoch}
Step: {step:,} / {total_steps:,}
Loss: {loss:.4f}
Accuracy: {acc:.4f}
        """
        
        return Panel(
            content.strip(),
            title="[bold green]Training Progress[/bold green]",
            border_style="bright_blue"
        )