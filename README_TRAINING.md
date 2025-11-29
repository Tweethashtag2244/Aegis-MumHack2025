# CNN Model Training Guide

## Overview

This guide explains how to train the CNN model for video artifact detection and integrate it with the perceptual forensics agent.

## Prerequisites

1. **Dataset**: You need a dataset with:
   - Real videos (camera-captured footage)
   - AI-generated videos (from Sora, Runway, Pika, Synthesia, etc.)

2. **Python Dependencies**: 
   - TensorFlow 2.15.0
   - OpenCV
   - NumPy
   - scikit-learn

## Dataset Structure

Organize your dataset as follows:

```
data/
├── real_videos/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── ai_videos/
    ├── sora_video1.mp4
    ├── runway_video1.mp4
    └── ...
```

## Training Steps

### 1. Prepare Your Dataset

Place your real videos in `data/real_videos/` and AI-generated videos in `data/ai_videos/`.

**Recommended dataset size:**
- Minimum: 50-100 videos per class
- Ideal: 500+ videos per class
- More data = better model performance

### 2. Run Training Script

Basic usage:
```bash
python train_cnn_model.py --real-dir data/real_videos --ai-dir data/ai_videos
```

With custom options:
```bash
python train_cnn_model.py \
    --real-dir data/real_videos \
    --ai-dir data/ai_videos \
    --output models/artifact_detector.h5 \
    --epochs 100 \
    --batch-size 32 \
    --max-frames 15 \
    --sample-every 30
```

### 3. Training Parameters

- `--real-dir`: Directory containing real video files
- `--ai-dir`: Directory containing AI-generated video files
- `--output`: Path to save trained model (default: `models/artifact_detector.h5`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 32)
- `--max-frames`: Maximum frames to extract per video (default: 10)
- `--sample-every`: Sample every Nth frame (default: 30)
- `--no-augmentation`: Disable data augmentation

### 4. Monitor Training

The script will:
- Extract frames from videos
- Split into train/validation sets
- Apply data augmentation (brightness, contrast, rotation)
- Train the model with early stopping
- Save the best model based on validation loss
- Display training metrics and classification report

### 5. Model Output

After training, you'll get:
- `models/artifact_detector.h5` - The trained model
- `models/artifact_detector_history.json` - Training history (loss, accuracy, etc.)

## Using the Trained Model

### Automatic Detection

The `perceptual_forensics_agent.py` will automatically:
1. Check for `models/artifact_detector.h5` on startup
2. If found, use the trained CNN model
3. If not found, fall back to heuristic method

### Manual Model Path

You can specify a custom model path by modifying the `_model_path` variable in `perceptual_forensics_agent.py`:

```python
_model_path = 'models/artifact_detector.h5'  # Change this path
```

## Model Architecture

The CNN model uses:
- Input: 128x128 grayscale frames
- Architecture:
  - 3 Conv2D layers with BatchNormalization
  - MaxPooling and Dropout for regularization
  - Dense layers for classification
  - Output: Binary classification (0 = real, 1 = AI-generated)

## Training Tips

1. **Dataset Balance**: Ensure roughly equal numbers of real and AI videos
2. **Frame Sampling**: Adjust `--sample-every` based on video length
3. **Augmentation**: Keep enabled unless you have a very large dataset
4. **Early Stopping**: Model will stop if validation loss doesn't improve for 10 epochs
5. **GPU**: Use GPU for faster training (automatically detected by TensorFlow)

## Evaluation

After training, check:
- **Validation Accuracy**: Should be > 70% for a useful model
- **Precision/Recall**: Balance between false positives and false negatives
- **Confusion Matrix**: Understand model's strengths and weaknesses

## Troubleshooting

### "No training data available"
- Check that video directories exist and contain video files
- Verify video file formats are supported (.mp4, .avi, .mov, etc.)

### Low accuracy
- Increase dataset size
- Try different hyperparameters (learning rate, batch size)
- Check for dataset imbalance
- Verify video quality

### Out of memory
- Reduce `--batch-size`
- Reduce `--max-frames`
- Use smaller input resolution (modify model architecture)

## Integration with Existing System

Once trained, the model integrates seamlessly:
- No code changes needed in the Flask server
- Automatic fallback to heuristic if model unavailable
- Same API interface

## Example Training Session

```bash
# Create directories
mkdir -p data/real_videos data/ai_videos models

# Download/place your videos in the directories

# Train model
python train_cnn_model.py \
    --real-dir data/real_videos \
    --ai-dir data/ai_videos \
    --epochs 50 \
    --batch-size 32

# Output:
# Loading real videos...
#   Processed 10 real videos...
#   ...
# Loading AI-generated videos...
#   Processed 10 AI videos...
#   ...
# 
# Dataset Summary:
#   Total frames: 2000
#   Real frames: 1000
#   AI frames: 1000
# 
# Training model...
# Epoch 1/50
# ...
# ✓ Model saved to: models/artifact_detector.h5
```

## Next Steps

After training:
1. Restart the Flask server to load the new model
2. Test with sample videos
3. Monitor performance in production
4. Retrain periodically with new data

