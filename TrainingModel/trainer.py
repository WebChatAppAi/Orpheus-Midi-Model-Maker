"""
Melody Transformer Trainer
Exact implementation from the original notebook with CLI enhancements
"""

import os
import sys
import pickle
import random
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import gc
from pathlib import Path
from tqdm import tqdm
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

# Add tegridy-tools path
project_root = Path(__file__).parent.parent
tegridy_path = project_root / "tegridy-tools" / "tegridy-tools"
xtransformer_path = tegridy_path / "X-Transformer"
sys.path.append(str(tegridy_path))
sys.path.append(str(xtransformer_path))

import TMIDIX
from x_transformer_2_3_1 import TransformerWrapper, Decoder, AutoregressiveWrapper

class MelodyTrainer:
    def __init__(self, config, console):
        self.config = config
        self.console = console
        self.setup_training_params()
        
    def setup_training_params(self):
        """Setup training parameters from config"""
        # Model parameters
        self.SEQ_LEN = self.config['model']['seq_len']
        self.PAD_IDX = 18819  # Fixed vocabulary
        
        # Training parameters (EXACT from notebook)
        self.VALIDATE_EVERY = self.config['training'].get('validate_every', 100)
        self.SAVE_EVERY = self.config['training'].get('save_every', 500)
        self.GENERATE_EVERY = self.config['training'].get('generate_every', 250)
        self.GENERATE_LENGTH = self.config['training'].get('generate_length', 512)
        self.PRINT_STATS_EVERY = self.config['training'].get('print_stats_every', 10)
        
        self.NUM_EPOCHS = self.config['training']['num_epochs']
        self.BATCH_SIZE = self.config['training']['batch_size']
        self.GRADIENT_ACCUMULATE_EVERY = self.config['training']['gradient_accumulate_every']
        self.LEARNING_RATE = self.config['training']['learning_rate']
        self.GRAD_CLIP = self.config['training'].get('grad_clip', 1.0)
        
        # Output directories
        self.model_dir = Path(self.config['output']['model_dir'])
        self.sample_dir = Path(self.config['output']['sample_dir'])
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.nsteps = 0
    
    def load_dataset(self):
        """Load training dataset from pickle files"""
        self.console.print("📂 Loading dataset...")
        
        from config import ConfigManager
        config_manager = ConfigManager()
        pickle_files = config_manager.get_dataset_files(self.config['dataset']['path'])
        
        self.console.print(f"Found {len(pickle_files)} pickle files")
        
        train_data = set()
        chunks_counter = 0
        
        gc.disable()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Loading files...", total=len(pickle_files))
            
            for file_path in pickle_files:
                try:
                    with open(file_path, 'rb') as f:
                        train_d = pickle.load(f)
                    
                    for t in train_d:
                        if 0 <= max(t) < self.PAD_IDX:  # EXACT data integrity check from notebook
                            train_data.add(tuple(t))
                            chunks_counter += 1
                        else:
                            self.console.print('Bad data!!!')
                            
                except Exception as e:
                    self.console.print(f"Error loading {file_path}: {e}")
                
                progress.advance(task)
        
        gc.enable()
        gc.collect()
        
        self.train_data = list(train_data)
        
        self.console.print(f"✅ Loaded {chunks_counter} total chunks")
        self.console.print(f"✅ {len(self.train_data)} unique sequences after deduplication")
        
        # Randomize train data (EXACT from notebook)
        random.shuffle(self.train_data)
    
    def setup_model(self):
        """Setup model architecture (EXACT from notebook)"""
        self.console.print("🤖 Setting up model...")
        
        # Instantiate the model (EXACT architecture from notebook)
        self.model = TransformerWrapper(
            num_tokens = self.PAD_IDX + 1,
            max_seq_len = self.SEQ_LEN,
            attn_layers = Decoder(
                dim = self.config['model']['dim'],
                depth = self.config['model']['depth'],
                heads = self.config['model']['heads'],
                rotary_pos_emb = True,
                attn_flash = True,
            )
        )
        
        self.model = AutoregressiveWrapper(self.model, ignore_index=self.PAD_IDX, pad_value=self.PAD_IDX)
        self.model.cuda()
        
        # Display model info
        total_params = sum(p.numel() for p in self.model.parameters())
        self.console.print(f"✅ Model created: {total_params:,} parameters")
        
        # Setup optimizer and scaler (EXACT from notebook)
        self.dtype = torch.bfloat16
        self.ctx = torch.amp.autocast(device_type='cuda', dtype=self.dtype)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        self.scaler = torch.amp.GradScaler('cuda')
    
    def get_train_data_batch(self, tdata, index, seq_len, batch_size, pad_idx):
        """Data loader function (EXACT from notebook)"""
        batch = tdata[(index*batch_size):(index*batch_size)+batch_size]
        
        padded_batch = []
        
        for ba in batch:
            ba = list(ba)
            
            if len(ba) > (seq_len+1):
                ba = ba[:(seq_len+1)]
            else:
                ba += [pad_idx] * ((seq_len+1) - len(ba[:(seq_len+1)]))
            
            padded_batch.append(ba)
        
        return torch.LongTensor(padded_batch).cuda()
    
    def generate_sample(self, step):
        """Generate sample MIDI (EXACT from notebook)"""
        self.model.eval()
        
        inp = random.choice(self.get_train_data_batch(
            self.train_data, step, self.SEQ_LEN, self.BATCH_SIZE, self.PAD_IDX
        ))[:self.GENERATE_LENGTH]
        
        with self.ctx:
            sample = self.model.generate(inp[None, ...], self.GENERATE_LENGTH)
        
        data = sample.tolist()[0]
        
        if len(data) != 0:
            # EXACT MIDI conversion from notebook
            song = data
            song_f = []
            
            time = 0
            dur = 1
            vel = 90
            pitch = 60
            channel = 0
            patch = 0
            
            patches = [-1] * 16
            channels = [0] * 16
            channels[9] = 1
            
            for ss in song:
                if 0 <= ss < 256:
                    time += ss * 16
                
                if 256 <= ss < 16768:
                    patch = (ss-256) // 128
                    
                    if patch < 128:
                        if patch not in patches:
                            if 0 in channels:
                                cha = channels.index(0)
                                channels[cha] = 1
                            else:
                                cha = 15
                            patches[cha] = patch
                            channel = patches.index(patch)
                        else:
                            channel = patches.index(patch)
                    
                    if patch == 128:
                        channel = 9
                    
                    pitch = (ss-256) % 128
                
                if 16768 <= ss < 18816:
                    dur = ((ss-16768) // 8) * 16
                    vel = (((ss-16768) % 8)+1) * 15
                    song_f.append(['note', time, dur, channel, pitch, vel])
            
            patches = [0 if x==-1 else x for x in patches]
            
            # Save sample MIDI
            sample_filename = self.sample_dir / f'sample_step_{self.nsteps}'
            TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(
                song_f,
                output_signature='ALVAN Music Transformer',
                output_file_name=str(sample_filename),
                track_name='ALVAN Generated Sample',
                list_of_MIDI_patches=patches
            )
            
            self.console.print(f"🎵 Sample generated: {sample_filename}.mid")
        
        self.model.train()
    
    def save_checkpoint(self, checkpoint_name=None):
        """Save model checkpoint"""
        if checkpoint_name is None:
            checkpoint_name = f'checkpoint_{self.nsteps}_steps_{self.train_losses[-1]:.4f}_loss_{self.train_accs[-1]:.4f}_acc.pth'
        
        checkpoint_path = self.model_dir / checkpoint_name
        
        torch.save(self.model.state_dict(), checkpoint_path)
        torch.save(self.optim.state_dict(), self.model_dir / 'optimizer.pth')
        torch.save(self.scaler.state_dict(), self.model_dir / 'scaler.pth')
        
        # Save training progress
        progress_data = [self.train_losses, self.train_accs, self.val_losses, self.val_accs]
        with open(self.model_dir / 'training_progress.pkl', 'wb') as f:
            pickle.dump(progress_data, f)
        
        self.console.print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def plot_training_progress(self):
        """Plot training progress"""
        if len(self.train_losses) > 0:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.train_losses, 'b-', label='Training Loss')
            if self.val_losses:
                val_steps = [i * self.VALIDATE_EVERY for i in range(len(self.val_losses))]
                plt.plot(val_steps, self.val_losses, 'r-', label='Validation Loss')
            plt.title('Training Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(self.train_accs, 'g-', label='Training Accuracy')
            if self.val_accs:
                val_steps = [i * self.VALIDATE_EVERY for i in range(len(self.val_accs))]
                plt.plot(val_steps, self.val_accs, 'orange', label='Validation Accuracy')
            plt.title('Training Accuracy')
            plt.xlabel('Steps')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(self.model_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def train(self):
        """Main training loop (EXACT from notebook)"""
        # Load dataset and setup model
        self.load_dataset()
        self.setup_model()
        
        # Calculate batches
        NUM_BATCHES = len(self.train_data) // self.BATCH_SIZE // self.GRADIENT_ACCUMULATE_EVERY
        total_steps = NUM_BATCHES * self.NUM_EPOCHS
        
        self.console.print(f"\n🚀 Starting Training")
        self.console.print(f"📊 Total steps: {total_steps:,}")
        self.console.print(f"📊 Batches per epoch: {NUM_BATCHES:,}")
        
        # Training loop (EXACT from notebook)
        for ep in range(self.NUM_EPOCHS):
            
            self.console.print(f"\n📚 Epoch {ep + 1}/{self.NUM_EPOCHS}")
            
            # Randomize train data (EXACT from notebook)
            random.shuffle(self.train_data)
            
            self.model.train()
            
            # Progress bar for epoch
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console
            ) as progress:
                
                epoch_task = progress.add_task(f"Epoch {ep+1}", total=NUM_BATCHES)
                
                for i in range(NUM_BATCHES):
                    
                    # EXACT training step from notebook
                    self.optim.zero_grad()
                    
                    for j in range(self.GRADIENT_ACCUMULATE_EVERY):
                        with self.ctx:
                            loss, acc = self.model(self.get_train_data_batch(
                                self.train_data, 
                                (i*self.GRADIENT_ACCUMULATE_EVERY)+j, 
                                self.SEQ_LEN, 
                                self.BATCH_SIZE, 
                                self.PAD_IDX
                            ))
                        loss = loss / self.GRADIENT_ACCUMULATE_EVERY
                        self.scaler.scale(loss).backward()
                    
                    # Store metrics
                    self.train_losses.append(loss.item() * self.GRADIENT_ACCUMULATE_EVERY)
                    self.train_accs.append(acc.item())
                    
                    # Optimization step
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.GRAD_CLIP)
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    
                    self.nsteps += 1
                    
                    # Print stats (EXACT from notebook)
                    if i % self.PRINT_STATS_EVERY == 0:
                        progress.console.print(
                            f"Step {self.nsteps}: Loss={self.train_losses[-1]:.4f}, Acc={self.train_accs[-1]:.4f}"
                        )
                    
                    # Validation (EXACT from notebook)
                    if i % self.VALIDATE_EVERY == 0:
                        self.model.eval()
                        with torch.no_grad():
                            with self.ctx:
                                val_loss, val_acc = self.model(self.get_train_data_batch(
                                    self.train_data, i, self.SEQ_LEN, self.BATCH_SIZE, self.PAD_IDX
                                ))
                                
                                self.val_losses.append(val_loss.item())
                                self.val_accs.append(val_acc.item())
                                
                                progress.console.print(
                                    f"📈 Validation - Loss: {val_loss.item():.4f}, Acc: {val_acc.item():.4f}"
                                )
                                
                                # Plot progress
                                self.plot_training_progress()
                        
                        self.model.train()
                    
                    # Generate sample (EXACT from notebook)
                    if i % self.GENERATE_EVERY == 0:
                        self.generate_sample(i)
                    
                    # Save checkpoint (EXACT from notebook)
                    if i % self.SAVE_EVERY == 0:
                        self.save_checkpoint()
                    
                    # Memory cleanup
                    if i % 10 == 0:
                        torch.cuda.empty_cache()
                    
                    progress.advance(epoch_task)
        
        # Final save
        self.save_checkpoint("final_model.pth")
        self.plot_training_progress()
        
        self.console.print("🎉 Training completed!")
