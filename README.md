<div align="center">
  <img src="https://raw.githubusercontent.com/WebChatAppAi/Orpheus-Midi-Model-Maker/main/image1.png" alt="Orpheus MIDI Model Maker Showcase" width="700"/>
  <h1>Orpheus MIDI Model Maker</h1>
  <p>
    <b>The easiest way to create AI melody generators from your own MIDI files!</b>
  </p>
  <p>
    Got a folder of MIDI files? Turn them into a personalized AI model that composes new melodies just for you. No coding, no complicated setup—just your music, your style, your AI.
  </p>
</div>

---

## ✨ Features

- **🎵 **One-Click Start**: Launch a simple, friendly command-line interface with `python app.py`.
- **📂 **Effortless File Finding**: Just point to a folder—our tool finds **all** MIDI files, even in subdirectories.
- **⚙️ **Smart Auto-Configuration**: Automatically detects your hardware (GPU/CPU) and suggests the best settings for you.
- **🤖 **Train, Don't Tweak**: The app generates a ready-to-use configuration file. No more manual setup unless you want to dive deep.
- **📊 **Live Dashboard**: Watch your model learn in real-time with a beautiful, live-updating dashboard.
- **🎹 **Test Your Creations**: A clear path to use your trained model in a fun, interactive piano roll application.

---

## 🚀 Quick Start: Your First AI Model in Minutes

Let's say you have a folder named `MyMIDIs` on your computer, filled with your favorite MIDI tracks. Here’s how you turn them into an AI model:

### 1. **Setup**

Clone the repository and install the necessary packages. This gets the tool ready.

```bash
git clone https://github.com/WebChatAppAi/Orpheus-Midi-Model-Maker
cd Orpheus-Midi-Model-Maker
pip install -r requirements.txt
```

### 2. **Launch the App**

Run the main application.

```bash
python app.py
```

### 3. **Create Your Dataset**

1.  From the main menu, choose **`Manage Datasets`**, then **`Create a New Dataset`**.
2.  When asked for the path to your MIDI files, just enter the path to your `MyMIDIs` folder.
    - *The tool will scan every subfolder inside `MyMIDIs` to find all `.mid` files.*
3.  Give your dataset a simple name (e.g., `my-awesome-melodies`).

The tool will process the files and prepare them for training.

### 4. **Train Your Model**

1.  Go back to the main menu and select **`Manage Models`**, then **`Start a New Training Session`**.
2.  The app will automatically create an optimal configuration (`.yml` file) for you. Just press **Enter** to use it.
3.  The training begins! Your live dashboard will appear, showing you the progress.

Once training is complete, your final AI model will be saved as a `.pth` file inside the `TrainingModel/models/` directory.

---

## 🎹 Using Your Trained AI Model

The training process creates a powerful AI model file (`.pth`), not just sample MIDI files. To use this model and generate new music, you can load it into our **MIDI-Gen Piano Roll** application.

**➡️ [Use Your Model with MIDI-Gen Piano Roll](https://github.com/WebChatAppAi/midi-gen)**

<a href="https://github.com/WebChatAppAi/midi-gen">
  <img src="https://user-images.githubusercontent.com/125438147/285415912-55565510-5022-445b-8b02-354a6b0c0612.png" alt="MIDI-Gen Piano Roll" width="500"/>
</a>

**How it works:**
1.  Find your trained model in `TrainingModel/models/your_model.pth`.
2.  Launch the MIDI-Gen Piano Roll application.
3.  Load your `.pth` file into the app to start generating and listening to new melodies created by your very own AI!

*(The MIDI files saved in the `samples` folder during training are just for a quick preview to see how the training is progressing.)*

---

## 📂 Project Structure

```
Orpheus-Midi-Model-Maker/
├── app.py                # The only file you need to run!
├── cli/                  # Modules for the command-line interface
├��─ DATA/                 # Your processed datasets live here
├── DatasetCreation/      # Scripts for processing MIDI files
└── TrainingModel/        # Scripts for training the AI model
    ├── models/           # Your final .pth models are saved here!
    ├── samples/          # MIDI samples generated during training
    └── sample_config.yml # An example configuration file
```

---

## 💡 GPU Memory Guide

The app auto-configures this, but if you want to customize your `.yml` file manually, here’s a guide:

| GPU Memory | Recommended Settings |
|------------|---------------------|
| 8GB        | `seq_len: 512`, `dim: 768`, `batch_size: 1` |
| 12GB       | `seq_len: 1024`, `dim: 1024`, `batch_size: 2` |
| 16GB       | `seq_len: 2048`, `dim: 1280`, `batch_size: 4` |
| 24GB+      | `seq_len: 4096`, `dim: 1536`, `batch_size: 8` |

---

## 🚨 What This Project Does NOT Do

- ❌ Generate full arrangements (drums, bass, etc.). It focuses purely on melody.
- ❌ Handle audio files (like `.mp3` or `.wav`). It's MIDI-only.
- ❌ Work without an NVIDIA GPU for training.

---

## 🤝 Contributing

This project is built on the amazing work of the open-source community. A special thanks to the creators of **[tegridy-tools](https://github.com/asigalov61/tegridy-tools)** and **[X-Transformer](https://github.com/lucidrains/x-transformers)**.

Contributions are welcome! Please open an issue or submit a pull request.

---

<div align="center">
  <b>🎵 Happy Music Making! 🎵</b>
</div>