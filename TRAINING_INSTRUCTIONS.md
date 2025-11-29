# Training Instructions - Windows

## Your Dataset Paths

Based on your setup:
- **Real Videos**: `C:\Users\kshau\Downloads\archive\real`
- **AI Videos**: `C:\Users\kshau\Downloads\archive\ai` (create this if needed)

## Quick Start

### Option 1: Use the PowerShell Script (Recommended for PowerShell)

1. **Create AI videos directory** (if you have AI videos):
   ```powershell
   mkdir C:\Users\kshau\Downloads\archive\ai
   ```
   Then copy your AI-generated videos there.

2. **Run the PowerShell training script**:
   ```powershell
   .\train_model.ps1
   ```
   
   If you get an execution policy error, run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\train_model.ps1
   ```

### Option 2: Use the Batch Script (For CMD)

1. **Create AI videos directory** (if you have AI videos):
   ```cmd
   mkdir C:\Users\kshau\Downloads\archive\ai
   ```
   Then copy your AI-generated videos there.

2. **Run the batch training script**:
   ```cmd
   train_model.bat
   ```

### Option 3: Manual Command (PowerShell)

**Single line (easiest)**:
```powershell
python train_cnn_model.py --real-dir "C:\Users\kshau\Downloads\archive\real" --ai-dir "C:\Users\kshau\Downloads\archive\ai"
```

**Multi-line (use backticks `)**:
```powershell
python train_cnn_model.py `
    --real-dir "C:\Users\kshau\Downloads\archive\real" `
    --ai-dir "C:\Users\kshau\Downloads\archive\ai"
```

### Option 4: Custom Paths

If your AI videos are in a different location:

```powershell
python train_cnn_model.py --real-dir "C:\Users\kshau\Downloads\archive\real" --ai-dir "C:\path\to\your\ai\videos"
```

## Full Training Command with All Options

### For PowerShell (use backticks `):
```powershell
python train_cnn_model.py `
    --real-dir "C:\Users\kshau\Downloads\archive\real" `
    --ai-dir "C:\Users\kshau\Downloads\archive\ai" `
    --output "models\artifact_detector.h5" `
    --epochs 50 `
    --batch-size 32 `
    --max-frames 10 `
    --sample-every 30
```

### For CMD (use caret ^):
```cmd
python train_cnn_model.py ^
    --real-dir "C:\Users\kshau\Downloads\archive\real" ^
    --ai-dir "C:\Users\kshau\Downloads\archive\ai" ^
    --output "models\artifact_detector.h5" ^
    --epochs 50 ^
    --batch-size 32 ^
    --max-frames 10 ^
    --sample-every 30
```

### Single Line (works in both):
```powershell
python train_cnn_model.py --real-dir "C:\Users\kshau\Downloads\archive\real" --ai-dir "C:\Users\kshau\Downloads\archive\ai" --output "models\artifact_detector.h5" --epochs 50 --batch-size 32 --max-frames 10 --sample-every 30
```

## Parameters Explained

- `--real-dir`: Your real videos directory (already set to your path)
- `--ai-dir`: Your AI videos directory (default: `C:\Users\kshau\Downloads\archive\ai`)
- `--output`: Where to save the trained model (default: `models\artifact_detector.h5`)
- `--epochs`: Number of training iterations (default: 50)
- `--batch-size`: Number of samples per batch (default: 32)
- `--max-frames`: Max frames to extract per video (default: 10)
- `--sample-every`: Extract every Nth frame (default: 30)

## Directory Structure

Your directories should look like:

```
C:\Users\kshau\Downloads\archive\
├── real\
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── ai\          (create this if needed)
    ├── sora_video1.mp4
    ├── runway_video1.mp4
    └── ...
```

## Supported Video Formats

The script supports:
- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.webm`
- `.flv`

## After Training

Once training completes:

1. **Model will be saved to**: `models\artifact_detector.h5`

2. **Restart the Flask server** to use the trained model:
   ```cmd
   python perceptual_forensics_agent.py
   ```

3. The system will automatically:
   - Detect the trained model
   - Use it instead of the heuristic method
   - Show "trained CNN model" in the analysis output

## Troubleshooting

### "No training data available"
- Check that `C:\Users\kshau\Downloads\archive\real` contains video files
- Verify video file extensions are supported
- Check file permissions

### "AI videos directory not found"
- Create the directory: `mkdir C:\Users\kshau\Downloads\archive\ai`
- Or specify a different path with `--ai-dir`

### Out of Memory
- Reduce batch size: `--batch-size 16`
- Reduce frames per video: `--max-frames 5`

### Slow Training
- Use GPU if available (TensorFlow will auto-detect)
- Reduce `--max-frames` to extract fewer frames per video
- Increase `--sample-every` to skip more frames

## Example Training Session

```powershell
PS C:\Users\kshau\Downloads\Aegis-MumHack2025-main\Aegis-MumHack2025-main> python train_cnn_model.py --real-dir "C:\Users\kshau\Downloads\archive\real" --ai-dir "C:\Users\kshau\Downloads\archive\ai"

============================================================
AEGIS CNN Model Training
============================================================

Loading real videos...
  Processed 10 real videos...
  Processed 20 real videos...
  ...

Loading AI-generated videos...
  Processed 10 AI videos...
  ...

Dataset Summary:
  Total frames: 2000
  Real frames: 1000
  AI frames: 1000
  Real videos processed: 50
  AI videos processed: 50

Splitting dataset...
  Training samples: 1600
  Validation samples: 400

Training model...
Epoch 1/50
...
✓ Model saved to: models/artifact_detector.h5
```

## Next Steps

After successful training:
1. ✅ Model is saved to `models\artifact_detector.h5`
2. ✅ Restart Flask server: `python perceptual_forensics_agent.py`
3. ✅ Test with sample videos in the web app
4. ✅ Check that analysis shows "trained CNN model" instead of "heuristic analysis"

