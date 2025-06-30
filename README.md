# 🎵 MidiModel-Creator

**Create AI melody generators from your MIDI files!**

Transform your collection of MIDI files into a trained AI model that can generate new melodies in the same style. Perfect for musicians, composers, and AI enthusiasts who want to create personalized music generation models.

## ✨ What This Project Does

**MidiModel-Creator** is a complete pipeline that:

1. **📁 Processes MIDI Files** → Converts your MIDI collection into AI-ready training data
2. **🤖 Trains AI Models** → Creates transformer-based melody generation models  
3. **🎵 Generates New Music** → Produces original melodies in your musical style

## 🎯 Key Features

### 🔧 **Dataset Creation**
- **Smart MIDI Processing**: Optimized for melody extraction from any MIDI file
- **Small File Support**: Works with short melodies (as few as 8 notes)
- **Quality Filtering**: Automatically removes poor-quality or corrupted files
- **Multi-instrument Handling**: Extracts melody lines from complex arrangements

### 🚀 **Advanced Training**
- **Beautiful CLI Interface**: Professional progress tracking and monitoring
- **GPU Optimized**: Automatic memory optimization for different GPU sizes
- **Real-time Monitoring**: Live loss/accuracy graphs and training statistics
- **Sample Generation**: Creates MIDI samples during training to monitor quality
- **Auto Checkpoints**: Never lose training progress with automatic saves

### ⚙️ **User-Friendly**
- **No Coding Required**: Simple setup and configuration files
- **Flexible Configuration**: YAML-based settings for easy customization
- **Global Ready**: Works on any Linux system with CUDA GPU

### 🎵 **New! Orpheus CLI Interface**
- **Unified Experience**: Single entry point for all features
- **Smart Configuration**: Auto-detects GPU and suggests optimal settings
- **Live Dashboard**: Real-time training monitoring with progress bars
- **Beautiful UI**: Professional terminal interface with Rich library
- **Guided Workflow**: Step-by-step guidance through the entire process

## 📋 Requirements

- **OS**: Linux (Ubuntu recommended)
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **Python**: 3.8+
- **Storage**: ~5GB for dependencies + space for your MIDI files

## 🚀 Quick Start

### 1. **Setup**
```bash
git clone https://github.com/your-repo/MidiModel-Creator
cd MidiModel-Creator
bash setup.sh
pip install -r cli/requirements.txt  # Install CLI dependencies
```

### 2. **Run Unified CLI (Recommended)**
```bash
python app.py
```

This launches the **Orpheus Midi Model Maker** - a beautiful unified interface that guides you through:
- 🎼 Creating datasets from MIDI files
- ✅ Validating your datasets
- 🚀 Training with auto-configured settings
- 📊 Live training dashboard with real-time metrics

### Alternative: Use Individual Scripts

#### Create Dataset
```bash
cd DatasetCreation
python start.py
# Choose "Create New Dataset"
# Provide path to your MIDI files
```

#### Train Model
```bash
cd ../TrainingModel
python app.py
# Provide path to your training config YAML
```

### 4. **Generate Music**
Your trained model will automatically generate sample MIDI files during training!

## 📂 Project Structure

```
MidiModel-Creator/
├── app.py                # New! Unified CLI entry point
├── cli/                  # New! CLI interface modules
│   ├── menu.py          # Main menu system
│   ├── dataset_cli.py   # Dataset management
│   ├── training_cli.py  # Training management
│   ├── config_generator.py # Auto-configuration
│   ├── dashboard.py     # Live training monitor
│   └── utils.py         # Shared utilities
├── setup.sh              # One-time setup script
├── DATA/                 # Processed datasets
├── DatasetCreation/      # MIDI processing tools
│   ├── start.py         # Main interface
│   ├── processdata.py   # MIDI processor
│   └── verifydata.py    # Dataset validator
└── TrainingModel/        # AI training module
    ├── app.py           # Training interface
    ├── config.py        # Configuration manager
    ├── trainer.py       # Training engine
    └── sample_config.yml # Example settings
```

## 🎵 What You Get

### **Input**: Your MIDI Files
- Classical compositions
- Jazz standards  
- Pop melodies
- Electronic music
- Any MIDI format

### **Output**: Trained AI Model
- Generates new melodies in your style
- Customizable length and creativity
- Professional MIDI output
- Ready for use in DAWs

## 🔧 Configuration

Create a YAML config file to customize training:

```yaml
# Dataset path
dataset:
  path: "./DATA"

# Model size (adjust for your GPU)
model:
  seq_len: 2048    # Context length
  dim: 1280        # Model size
  depth: 8         # Transformer layers
  heads: 20        # Attention heads

# Training settings
training:
  batch_size: 8              # Batch size
  gradient_accumulate_every: 6
  learning_rate: 0.0001
  num_epochs: 5

# Output paths
output:
  model_dir: "./models"
  sample_dir: "./samples"
```

## 💡 GPU Memory Guide

| GPU Memory | Recommended Settings |
|------------|---------------------|
| 8GB        | seq_len: 512, dim: 768, batch_size: 1 |
| 12GB       | seq_len: 1024, dim: 1024, batch_size: 2 |
| 16GB       | seq_len: 2048, dim: 1280, batch_size: 4 |
| 24GB+      | seq_len: 4096, dim: 1536, batch_size: 8 |

## 🎯 Use Cases

- **🎼 Composers**: Generate melodic ideas and variations
- **🎮 Game Developers**: Create dynamic background music
- **📚 Music Students**: Analyze and learn from different musical styles  
- **🤖 AI Researchers**: Experiment with music generation models
- **🎵 Musicians**: Explore new creative possibilities

## 🚨 What This Project Does NOT Do

- ❌ Generate full arrangements (drums, bass, harmony)
- ❌ Handle audio files (only MIDI)
- ❌ Real-time generation (training required first)
- ❌ Work without NVIDIA GPU

## 📊 Expected Results

### **Training Time**
- **Small dataset** (1,000 files): 2-4 hours
- **Medium dataset** (10,000 files): 8-12 hours  
- **Large dataset** (50,000+ files): 24-48 hours

### **Output Quality**
- **Beginner**: Recognizable melodies after 1 epoch
- **Good**: Musical coherence after 3 epochs
- **Excellent**: Style-accurate generation after 5 epochs

## 🛠️ Troubleshooting

**Out of Memory?** → Reduce `batch_size` and `seq_len` in config
**No MIDI files found?** → Check file extensions (.mid, .midi)
**Poor quality output?** → Increase training time or improve input data quality
**Training too slow?** → Increase `batch_size` if you have more GPU memory

## 🤝 Contributing

This project is built on:
- **[tegridy-tools](https://github.com/asigalov61/tegridy-tools)** - MIDI processing
- **[X-Transformer](https://github.com/lucidrains/x-transformers)** - Transformer architecture
- **PyTorch** - Deep learning framework

## 📄 License

[Add your chosen license here]

## 🎵 Happy Music Making!

Transform your MIDI collection into an AI composer and discover new musical possibilities!

---
*Made with ❤️ for the music and AI community*
