# 🎹 Piano Music Generator

A deep learning-based MIDI music generator trained on the MAESTRO dataset, capable of generating classical piano compositions.

![Header](images/music.jpg)
Image by <a href="https://pixabay.com/users/ralf1403-21380246/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=8463988">Ralf Ruppert</a> from <a href="https://pixabay.com//?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=8463988">Pixabay</a>

## 📋 Table of Contents

- [Project Setup & Data Exploration](#project-setup)
- [Data Preprocessing & Feature Engineering](#data-preprocessing)
- [Model Architecture Design](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Generation & Sampling](#generation-sampling)
- [Evaluation & Refinement](#evaluation-refinement)
- [Portfolio Presentation](#portfolio-presentation)

## 🚀 Getting Started

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset

This project uses the [MAESTRO v3.0.0 dataset](https://magenta.tensorflow.org/datasets/maestro), which contains over 200 hours of virtuosic piano performances captured with fine alignment (~3 ms) between note labels and audio.

## 📊 Project Structure

```
piano-music-generator/
├── data/                          # MAESTRO dataset
├── notebooks/                     # Jupyter notebooks for exploration and experiments
├── src/                          # Source code
│   ├── data/                     # Data processing utilities
│   ├── models/                   # Model architectures
│   ├── training/                 # Training scripts
│   └── generation/               # Music generation utilities
├── models/                       # Saved model checkpoints
├── generated_samples/            # Generated MIDI outputs
└── requirements.txt              # Project dependencies
```

## 🎵 Features

- MIDI data preprocessing and feature extraction
- Multiple model architectures (LSTM, Transformer, VAE)
- Temperature-based and nucleus sampling
- Music quality evaluation metrics
- Interactive generation interface

## 📝 License

MIT License - feel free to use this project for learning and portfolio purposes.

## 🙏 Acknowledgments

- MAESTRO Dataset by Google Magenta
- International Piano-e-Competition
