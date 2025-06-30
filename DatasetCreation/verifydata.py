"""
Dataset Verification Module
Validates existing processed datasets
"""

import os
import sys
import pickle
import glob
import statistics
from pathlib import Path

# Add tegridy-tools to path  
project_root = Path(__file__).parent.parent
tegridy_path = project_root / "tegridy-tools" / "tegridy-tools"
sys.path.append(str(tegridy_path))

def verify_dataset(dataset_path):
    """
    Verify existing dataset files
    
    Args:
        dataset_path (str): Path to dataset directory
    """
    
    print(f"\n✅ VERIFYING DATASET")
    print("=" * 30)
    print(f"📁 Dataset path: {dataset_path}")
    
    # Find pickle files
    pickle_files = []
    for ext in ['*.pkl', '*.pickle']:
        pattern = os.path.join(dataset_path, '**', ext)
        pickle_files.extend(glob.glob(pattern, recursive=True))
    
    if not pickle_files:
        print("❌ No pickle files found!")
        return False
    
    print(f"\n📊 Found {len(pickle_files)} pickle files:")
    
    total_sequences = 0
    valid_files = 0
    invalid_files = 0
    
    for file_path in pickle_files:
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / 1024  # KB
        
        print(f"\n📄 Checking: {file_name} ({file_size:.1f} KB)")
        
        try:
            # Load pickle file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Validate data structure
            if isinstance(data, list):
                # Dataset file
                if not data:
                    print("   ⚠️  Empty dataset")
                    continue
                
                # Check if it's a list of sequences
                if isinstance(data[0], (list, tuple)):
                    sequences = data
                    sequence_count = len(sequences)
                    total_sequences += sequence_count
                    
                    # Analyze sequences
                    sequence_lengths = [len(seq) for seq in sequences]
                    avg_length = sum(sequence_lengths) / len(sequence_lengths)
                    min_length = min(sequence_lengths)
                    max_length = max(sequence_lengths)
                    
                    # Check token validity
                    all_tokens = []
                    for seq in sequences[:100]:  # Sample first 100 sequences
                        all_tokens.extend(seq)
                    
                    if all_tokens:
                        min_token = min(all_tokens)
                        max_token = max(all_tokens)
                        
                        # Check if tokens are in valid range (0-18818)
                        if min_token >= 0 and max_token <= 18818:
                            token_status = "✅ Valid"
                        else:
                            token_status = f"❌ Invalid range: {min_token}-{max_token}"
                    else:
                        token_status = "⚠️  No tokens found"
                    
                    print(f"   🎵 Sequences: {sequence_count}")
                    print(f"   📏 Avg length: {avg_length:.1f} tokens")
                    print(f"   📏 Range: {min_length}-{max_length} tokens")
                    print(f"   🔢 Tokens: {token_status}")
                    
                    valid_files += 1
                
                else:
                    print("   ❌ Invalid format: Not a sequence dataset")
                    invalid_files += 1
            
            elif isinstance(data, dict):
                # Metadata file
                print("   📊 Metadata file:")
                for key, value in data.items():
                    if isinstance(value, dict):
                        print(f"      {key}: {value}")
                    else:
                        print(f"      {key}: {value}")
                valid_files += 1
            
            else:
                print(f"   ❌ Unknown data type: {type(data)}")
                invalid_files += 1
                
        except Exception as e:
            print(f"   ❌ Error loading file: {e}")
            invalid_files += 1
    
    # Summary
    print(f"\n📊 VERIFICATION SUMMARY")
    print("=" * 30)
    print(f"✅ Valid files: {valid_files}")
    print(f"❌ Invalid files: {invalid_files}")
    print(f"🎵 Total sequences: {total_sequences}")
    
    if valid_files > 0:
        print(f"✅ Dataset verification PASSED")
        return True
    else:
        print(f"❌ Dataset verification FAILED")
        return False

def verify_single_file(file_path):
    """
    Verify a single dataset file
    
    Args:
        file_path (str): Path to pickle file
    
    Returns:
        bool: True if valid, False otherwise
    """
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list) and data:
            # Check first sequence
            sample_seq = data[0]
            if isinstance(sample_seq, (list, tuple)):
                # Check token range
                tokens = list(sample_seq)
                if tokens and min(tokens) >= 0 and max(tokens) <= 18818:
                    return True
        
        return False
        
    except Exception:
        return False