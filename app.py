#!/usr/bin/env python3
"""
Orpheus Midi Model Maker
Main entry point for the unified CLI tool

This is a user-friendly AI training and dataset creation CLI tool built on top of
@asigalov61's original pipeline: https://github.com/asigalov61/tegridy-tools

Developed by: https://github.com/WebChatAppAi
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import CLI components
try:
    from cli import MainMenu
except ImportError as e:
    print(f"Error importing CLI modules: {e}")
    print("\nPlease ensure you have installed all requirements:")
    print("  pip install rich typer pyyaml")
    sys.exit(1)

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Orpheus Midi Model Maker - Create AI melody generators from your MIDI files",
        epilog="For more information, visit: https://github.com/WebChatAppAi"
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Orpheus Midi Model Maker v1.0'
    )
    
    # Parse arguments (for future expansion)
    args = parser.parse_args()
    
    # Run main menu
    try:
        menu = MainMenu()
        menu.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your installation and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 