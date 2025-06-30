"""
MIDI Dataset Processing Module
Optimized for melody generation with small MIDI files support
"""

import os
import sys
import pickle
import random
import statistics
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# Add tegridy-tools to path
project_root = Path(__file__).parent.parent
tegridy_path = project_root / "tegridy-tools" / "tegridy-tools"
sys.path.append(str(tegridy_path))

import TMIDIX

def melody_optimized_midi_processor(midi_file):
    """
    Process MIDI file optimized for melody extraction and small files
    
    Args:
        midi_file (str): Path to MIDI file
    
    Returns:
        list: Integer sequence representing the melody or None if processing fails
    """
    
    try:
        # Read MIDI file
        raw_score = TMIDIX.midi2single_track_ms_score(midi_file)
        
        # Advanced score processing
        escore_result = TMIDIX.advanced_score_processor(
            raw_score, 
            return_enhanced_score_notes=True, 
            apply_sustain=True
        )
        
        if not escore_result:
            return None
            
        escore_notes = escore_result[0]
        
        if not escore_notes or len(escore_notes) < 8:  # Very low minimum for small files
            return None
        
        # Augment score
        escore_notes = TMIDIX.augment_enhanced_score_notes(escore_notes, sort_drums_last=True)
        
        # Get instrument information
        instruments_list = sorted(set([y[6] for y in escore_notes]))
        instruments_list_without_drums = [i for i in instruments_list if i != 128]
        
        # Focus on melody extraction
        if instruments_list_without_drums:
            # Filter out drums
            escore_notes_no_drums = [e for e in escore_notes if e[3] != 9]
            
            if escore_notes_no_drums:
                # For melody focus: extract main melodic line
                if len(instruments_list_without_drums) > 1:
                    # Multi-instrument: extract melody (highest pitched notes at each time)
                    melody_notes = []
                    time_groups = {}
                    
                    # Group notes by time
                    for note in escore_notes_no_drums:
                        time = note[1]
                        if time not in time_groups:
                            time_groups[time] = []
                        time_groups[time].append(note)
                    
                    # Select highest pitch at each time point
                    for time in sorted(time_groups.keys()):
                        notes_at_time = time_groups[time]
                        highest_note = max(notes_at_time, key=lambda x: x[4])  # x[4] is pitch
                        melody_notes.append(highest_note)
                    
                    escore_notes = melody_notes
                else:
                    # Single instrument: use all notes
                    escore_notes = escore_notes_no_drums
        
        # Check minimum notes after melody extraction
        if len(escore_notes) < 8:
            return None
        
        # Basic quality checks (relaxed for melody)
        escore_notes_tones = sorted(set([e[4] % 12 for e in escore_notes]))
        
        # Require at least 2 different pitch classes (very relaxed)
        if len(escore_notes_tones) < 2:
            return None
        
        # Velocity adjustment
        escore_notes_velocities = [e[5] for e in escore_notes]
        avg_velocity = sum(escore_notes_velocities) / len(escore_notes_velocities)
        
        if avg_velocity < 64:
            TMIDIX.adjust_score_velocities(escore_notes, 124)
        
        # Convert to delta score
        dscore = TMIDIX.delta_score_notes(escore_notes)
        
        # Chordify score
        dcscore = TMIDIX.chordify_score([d[1:] for d in dscore])
        
        # Build integer sequence
        melody_sequence = [18816]  # Start token
        
        # Process each chord/note group
        for i, c in enumerate(dcscore):
            
            # Add outro marker for last 32 events (reduced for smaller files)
            if len(dcscore) - i == 32 and len(dcscore) > 64:
                melody_sequence.extend([18817])
            
            # Delta time
            delta_time = c[0][0]
            melody_sequence.append(delta_time)
            
            # Process each note in the chord/group
            for e in c:
                
                # Extract note properties
                dur = max(1, min(255, e[1]))  # Duration
                pat = max(0, min(128, e[5]))  # Patch/Instrument
                ptc = max(1, min(127, e[3])) # Pitch
                vel = max(8, min(127, e[4])) # Velocity
                
                # Convert velocity to 8 levels
                velocity = round(vel / 15) - 1
                velocity = max(0, min(7, velocity))
                
                # Encode as integers
                pat_ptc = (128 * pat) + ptc
                dur_vel = (8 * dur) + velocity
                
                melody_sequence.extend([pat_ptc + 256, dur_vel + 16768])
            
            # Limit sequence length for memory efficiency
            if len(melody_sequence) > 4096:  # Reduced from 8192
                break
        
        # End token
        melody_sequence.extend([18818])
        
        return melody_sequence
    
    except Exception as ex:
        print(f"❌ Error processing {os.path.basename(midi_file)}: {ex}")
        return None

def process_midi_dataset(midi_files):
    """
    Process a list of MIDI files and create training dataset
    
    Args:
        midi_files (list): List of MIDI file paths
    """
    
    print("\n🎵 PROCESSING MIDI DATASET")
    print("=" * 50)
    print(f"📁 Total files to process: {len(midi_files)}")
    print("🎯 Optimized for melody generation")
    print("=" * 50)
    
    processed_sequences = []
    success_count = 0
    error_count = 0
    
    # Process files with progress bar
    for file_path in tqdm(midi_files, desc="Processing MIDI files"):
        
        result = melody_optimized_midi_processor(file_path)
        
        if result:
            processed_sequences.append(result)
            success_count += 1
        else:
            error_count += 1
    
    print(f"\n📊 PROCESSING RESULTS")
    print("=" * 30)
    print(f"✅ Successfully processed: {success_count}")
    print(f"❌ Failed to process: {error_count}")
    print(f"📈 Success rate: {success_count/len(midi_files)*100:.1f}%")
    
    if not processed_sequences:
        print("❌ No sequences were successfully processed!")
        return
    
    # Remove duplicates
    print("\n🔄 Removing duplicates...")
    unique_sequences = list(set(tuple(seq) for seq in processed_sequences))
    unique_sequences = [list(seq) for seq in unique_sequences]
    
    print(f"📊 Unique sequences after deduplication: {len(unique_sequences)}")
    
    # Analyze sequences
    sequence_lengths = [len(seq) for seq in unique_sequences]
    avg_length = sum(sequence_lengths) / len(sequence_lengths)
    min_length = min(sequence_lengths)
    max_length = max(sequence_lengths)
    
    print(f"\n📏 SEQUENCE STATISTICS")
    print("=" * 30)
    print(f"📊 Average length: {avg_length:.1f} tokens")
    print(f"📊 Length range: {min_length} - {max_length} tokens")
    print(f"🎵 Sample sequence: {unique_sequences[0][:20]}...")
    
    # Save processed dataset
    output_path = project_root / "DATA" / "melody_dataset.pkl"
    
    print(f"\n💾 Saving dataset...")
    with open(output_path, 'wb') as f:
        pickle.dump(unique_sequences, f)
    
    # Save metadata
    metadata = {
        'total_files_processed': len(midi_files),
        'successful_files': success_count,
        'failed_files': error_count,
        'success_rate': success_count / len(midi_files),
        'unique_sequences': len(unique_sequences),
        'avg_sequence_length': avg_length,
        'min_sequence_length': min_length,
        'max_sequence_length': max_length,
        'vocabulary_size': 18819,
        'token_ranges': {
            'time_deltas': '0-255',
            'patch_pitch': '256-16767', 
            'duration_velocity': '16768-18815',
            'start_token': 18816,
            'outro_token': 18817,
            'end_token': 18818
        }
    }
    
    metadata_path = project_root / "DATA" / "melody_dataset_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Dataset saved: {output_path}")
    print(f"✅ Metadata saved: {metadata_path}")
    
    # Verify saved data
    print(f"\n🔍 Verifying saved dataset...")
    try:
        with open(output_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        if len(loaded_data) == len(unique_sequences):
            print("✅ Dataset verification passed")
        else:
            print("❌ Dataset verification failed - length mismatch")
            
    except Exception as e:
        print(f"❌ Dataset verification failed: {e}")
    
    print("\n🎉 DATASET CREATION COMPLETE!")
    print("=" * 40)
    print(f"📁 Location: {output_path}")
    print(f"🎵 Ready for melody model training!")
