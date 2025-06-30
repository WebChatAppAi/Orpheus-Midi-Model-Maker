<div align="center">
  
  ![Orpheus MIDI Model Maker](https://raw.githubusercontent.com/WebChatAppAi/Orpheus-Midi-Model-Maker/main/image1.png)
  
  # 🎼 Orpheus MIDI Model Maker
  
  ### *Transform your MIDI collection into personalized AI composers*
  
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/CUDA-Required-green?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
    <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
  </p>
  
  > 🚀 **No coding experience required** | 🎯 **Your music, your style, your AI** | 🔥 **Production-ready models**
  
</div>

<br>

## ✨ **What Makes This Special?**

<table>
<tr>
<td width="50%">

### 🎵 **One-Click Magic**
Launch with a single command and let our intuitive CLI guide you through everything.

### 📂 **Smart File Discovery** 
Drop in any folder structure - we'll find every MIDI file, even in nested directories.

### ⚙️ **Auto-Configuration**
Hardware detection and optimal settings configuration - no manual tweaks needed.

</td>
<td width="50%">

### 🤖 **Train, Don't Code**
Focus on your music while our engine handles the complex AI training pipeline.

### 📊 **Real-Time Dashboard**
Watch your model evolve with beautiful, live-updating training visualizations.

### 🎹 **Instant Deployment**
Export ready-to-use models for immediate music generation in our Piano Roll app.

</td>
</tr>
</table>

---

## 🚀 **Quick Start Guide**

> **Prerequisites**: Python 3.8+, NVIDIA GPU with CUDA support

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/WebChatAppAi/Orpheus-Midi-Model-Maker
cd Orpheus-Midi-Model-Maker
```

### **Step 2: Setup Dependencies**

First, clone the required tegridy-tools repository:

```bash
git clone https://github.com/asigalov61/tegridy-tools.git
```

Then install the Python requirements:

```bash
pip install -r requirements.txt
```

### **Step 3: Launch the Application**

```bash
python app.py
```

---

## 🎯 **Create Your First AI Model**

Let's say you have a folder named `MyMIDIs` filled with your favorite MIDI tracks. Here's how to transform them into an AI composer:

### **1. Create Your Dataset**

1.  From the main menu, choose **`Manage Datasets`**, then **`Create a New Dataset`**.
2.  When asked for the path to your MIDI files, just enter the path to your `MyMIDIs` folder.
    - *The tool will scan every subfolder inside `MyMIDIs` to find all `.mid` files.*
3.  Give your dataset a simple name (e.g., `my-awesome-melodies`).

The tool will process the files and prepare them for training.

### **2. Train Your Model**

1.  Go back to the main menu and select **`Manage Models`**, then **`Start a New Training Session`**.
2.  The app will automatically create an optimal configuration (`.yml` file) for you. Just press **Enter** to use it.
3.  The training begins! Your live dashboard will appear, showing you the progress.

Once training is complete, your final AI model will be saved as a `.pth` file inside the `TrainingModel/models/` directory.

---

## 🎹 **Using Your Trained AI Model**

The training process creates a powerful AI model file (`.pth`), not just sample MIDI files. To use this model and generate new music, you can load it into our **MIDI-Gen Piano Roll** application.

**➡️ [Use Your Model with MIDI-Gen Piano Roll](https://github.com/WebChatAppAi/midi-gen)**

<div align="center">
  <a href="https://github.com/WebChatAppAi/midi-gen">
    <img src="https://github.com/WebChatAppAi/midi-gen/blob/main/image2.png?raw=true" alt="MIDI-Gen Piano Roll" width="600" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
  </a>
</div>

### **How it works:**
1.  🎯 Find your trained model in `TrainingModel/models/your_model.pth`
2.  🚀 Launch the MIDI-Gen Piano Roll application
3.  🎵 Load your `.pth` file into the app to start generating and listening to new melodies created by your very own AI!

> 💡 *The MIDI files saved in the `samples` folder during training are just for a quick preview to see how the training is progressing.*

---

## 📂 **Project Structure**

```
Orpheus-Midi-Model-Maker/
├── 🚀 app.py                # Main application - your entry point!
├── 📁 cli/                  # Intuitive command-line interface modules
├── 💾 DATA/                 # Your processed datasets storage
├── 🔧 DatasetCreation/      # MIDI processing & preparation scripts
├── 🤖 tegridy-tools/        # Required dependency (cloned separately)
└── 🎯 TrainingModel/        # AI training engine
    ├── 🎼 models/           # Your final .pth models saved here!
    ├── 🎵 samples/          # MIDI samples generated during training
    └── ⚙️ sample_config.yml # Example configuration template
```

---

## 🔧 **GPU Memory Optimization Guide**

The app auto-configures optimal settings, but for manual customization of your `.yml` configuration:

<table align="center">
<tr>
<th>🎮 GPU Memory</th>
<th>⚡ Recommended Settings</th>
<th>🎯 Performance</th>
</tr>
<tr>
<td><strong>8GB</strong></td>
<td><code>seq_len: 512</code><br><code>dim: 768</code><br><code>batch_size: 1</code></td>
<td>🟡 Good</td>
</tr>
<tr>
<td><strong>12GB</strong></td>
<td><code>seq_len: 1024</code><br><code>dim: 1024</code><br><code>batch_size: 2</code></td>
<td>🟢 Better</td>
</tr>
<tr>
<td><strong>16GB</strong></td>
<td><code>seq_len: 2048</code><br><code>dim: 1280</code><br><code>batch_size: 4</code></td>
<td>🔥 Great</td>
</tr>
<tr>
<td><strong>24GB+</strong></td>
<td><code>seq_len: 4096</code><br><code>dim: 1536</code><br><code>batch_size: 8</code></td>
<td>⚡ Excellent</td>
</tr>
</table>

---

## ⚠️ **Important Limitations**

<div align="center">

| ❌ **What We DON'T Do** | ✅ **What We DO** |
|--------------------------|-------------------|
| Generate full arrangements (drums, bass, etc.) | Focus on beautiful melody generation |
| Handle audio files (`.mp3`, `.wav`) | Process MIDI files exclusively |
| Work without NVIDIA GPU | Require CUDA for optimal training |

</div>

---

## 🤝 **Contributing & Credits**

<div align="center">

### **Built on the Shoulders of Giants**

This project leverages incredible open-source work from:

[![tegridy-tools](https://img.shields.io/badge/tegridy--tools-🎵-blue?style=for-the-badge)](https://github.com/asigalov61/tegridy-tools)
[![X-Transformer](https://img.shields.io/badge/X--Transformer-🤖-green?style=for-the-badge)](https://github.com/lucidrains/x-transformers)

### **Want to Contribute?**

We welcome contributions! Feel free to:
- 🐛 Report bugs via [Issues](https://github.com/WebChatAppAi/Orpheus-Midi-Model-Maker/issues)
- 💡 Suggest features 
- 🔧 Submit pull requests
- 📖 Improve documentation

</div>

---

<div align="center">
  
  ### 🎵 **Happy Music Making!** 🎵
  
  <p>
    <img src="https://img.shields.io/badge/Made%20with-❤️-red?style=for-the-badge" alt="Made with Love">
    <img src="https://img.shields.io/badge/For-Musicians-purple?style=for-the-badge" alt="For Musicians">
  </p>
  
</div>