"""
MIDI Model Creator - Dataset Creation Tool
Main entry point for MIDI dataset processing and validation
"""

import os
import sys
import glob
from pathlib import Path

# Add tegridy-tools to path
project_root = Path(__file__).parent.parent
tegridy_path = project_root / "tegridy-tools" / "tegridy-tools"
sys.path.append(str(tegridy_path))

def check_environment():
    """Check if required dependencies are available"""
    print("🔍 Checking Environment...")
    print("=" * 50)
    
    # Check if tegridy-tools exists
    if not tegridy_path.exists():
        print("❌ tegridy-tools not found!")
        print("💡 Please run setup.sh first:")
        print("   bash setup.sh")
        return False
    
    # Check if DATA folder exists
    data_folder = project_root / "DATA"
    if not data_folder.exists():
        print("📁 Creating DATA folder...")
        data_folder.mkdir(exist_ok=True)
    
    # Test TMIDIX import
    try:
        import TMIDIX
        print("✅ TMIDIX module available")
    except ImportError:
        print("❌ Cannot import TMIDIX!")
        print("💡 Please check tegridy-tools installation")
        return False
    
    print("✅ Environment check passed")
    return True

def show_menu():
    """Display main menu options"""
    print("\n🎵 MIDI Model Creator - Dataset Tool")
    print("=" * 50)
    print("Choose an option:")
    print("1. 📁 Create New Dataset from MIDI files")
    print("2. ✅ Verify Existing Dataset")
    print("3. ❌ Exit")
    print("=" * 50)
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("⚠️  Please enter 1, 2, or 3")

def scan_midi_files(directory_path):
    """Scan directory and subdirectories for MIDI files"""
    print(f"🔍 Scanning: {directory_path}")
    print("=" * 50)
    
    if not os.path.exists(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        return []
    
    # Find all MIDI files recursively
    midi_extensions = ['*.mid', '*.midi', '*.MID', '*.MIDI']
    midi_files = []
    
    for ext in midi_extensions:
        pattern = os.path.join(directory_path, '**', ext)
        midi_files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates and sort
    midi_files = sorted(list(set(midi_files)))
    
    print(f"📊 Found {len(midi_files)} MIDI files")
    
    # Show directory breakdown
    directories = {}
    for file in midi_files:
        dir_name = os.path.dirname(file)
        rel_dir = os.path.relpath(dir_name, directory_path)
        directories[rel_dir] = directories.get(rel_dir, 0) + 1
    
    print("\n📂 Directory breakdown:")
    for dir_name, count in sorted(directories.items()):
        print(f"   {dir_name}: {count} files")
    
    return midi_files

def create_dataset_option():
    """Handle dataset creation option"""
    print("\n📁 CREATE NEW DATASET")
    print("=" * 30)
    
    # Get MIDI directory path
    while True:
        midi_path = input("Enter path to MIDI files directory: ").strip()
        if os.path.exists(midi_path):
            break
        print(f"❌ Directory not found: {midi_path}")
        retry = input("Try again? (y/n): ").strip().lower()
        if retry != 'y':
            return
    
    # Scan for MIDI files
    midi_files = scan_midi_files(midi_path)
    
    if not midi_files:
        print("❌ No MIDI files found!")
        return
    
    # Confirm processing
    print(f"\n🎯 Ready to process {len(midi_files)} MIDI files")
    confirm = input("Continue with processing? (y/n): ").strip().lower()
    
    if confirm == 'y':
        from processdata import process_midi_dataset
        process_midi_dataset(midi_files)
    else:
        print("❌ Processing cancelled")

def verify_dataset_option():
    """Handle dataset verification option"""
    print("\n✅ VERIFY EXISTING DATASET")
    print("=" * 30)
    
    # Get dataset directory path
    while True:
        dataset_path = input("Enter path to dataset files directory: ").strip()
        if os.path.exists(dataset_path):
            break
        print(f"❌ Directory not found: {dataset_path}")
        retry = input("Try again? (y/n): ").strip().lower()
        if retry != 'y':
            return
    
    from verifydata import verify_dataset
    verify_dataset(dataset_path)

def main():
    """Main application entry point"""
    print("🎵 MIDI Model Creator v1.0")
    print("========================")
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Main application loop
    while True:
        choice = show_menu()
        
        if choice == '1':
            create_dataset_option()
        elif choice == '2':
            verify_dataset_option()
        elif choice == '3':
            print("👋 Goodbye!")
            break
    
    print("\n✅ Thanks for using MIDI Model Creator!")

if __name__ == "__main__":
    main()
