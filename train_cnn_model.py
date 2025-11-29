# =========================================
# AEGIS CNN Model Training Script
# =========================================

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
from pathlib import Path
import argparse

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------------------
# 1. Model Architecture (same as original)
# -----------------------------------------
def build_tf_model():
    """Build a small CNN for artifact probability estimation."""
    model = models.Sequential([
        layers.Input(shape=(128, 128, 1)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[Accuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall')]
    )
    
    return model


# -----------------------------------------
# 2. Data Loading and Preprocessing
# -----------------------------------------
def extract_frames_from_video(video_path, max_frames_per_video=10, sample_every_n=30):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        max_frames_per_video: Maximum frames to extract per video
        sample_every_n: Sample every Nth frame
    
    Returns:
        List of preprocessed frames (128x128 grayscale, normalized)
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Warning: Could not open {video_path}")
        return frames
    
    frame_count = 0
    extracted = 0
    
    while extracted < max_frames_per_video:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % sample_every_n != 0:
            continue
        
        # Preprocess frame (same as inference)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128)).astype('float32') / 255.0
        frames.append(resized)
        extracted += 1
    
    cap.release()
    return frames


def load_dataset(real_videos_dir, ai_videos_dir, max_frames_per_video=10, sample_every_n=30):
    """
    Load dataset from directories containing real and AI-generated videos.
    
    Args:
        real_videos_dir: Directory containing real video files
        ai_videos_dir: Directory containing AI-generated video files
        max_frames_per_video: Maximum frames to extract per video
        sample_every_n: Sample every Nth frame
    
    Returns:
        X: Array of frames (n_samples, 128, 128, 1)
        y: Array of labels (0 = real, 1 = AI-generated)
    """
    X = []
    y = []
    
    # Supported video formats
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    
    print("Loading real videos...")
    real_count = 0
    if os.path.exists(real_videos_dir):
        for video_file in Path(real_videos_dir).rglob('*'):
            if video_file.suffix.lower() in video_extensions:
                frames = extract_frames_from_video(str(video_file), max_frames_per_video, sample_every_n)
                for frame in frames:
                    X.append(frame)
                    y.append(0)  # 0 = real
                real_count += 1
                if real_count % 10 == 0:
                    print(f"  Processed {real_count} real videos...")
    else:
        print(f"Warning: Real videos directory not found: {real_videos_dir}")
    
    print(f"\nLoading AI-generated videos...")
    ai_count = 0
    if os.path.exists(ai_videos_dir):
        for video_file in Path(ai_videos_dir).rglob('*'):
            if video_file.suffix.lower() in video_extensions:
                frames = extract_frames_from_video(str(video_file), max_frames_per_video, sample_every_n)
                for frame in frames:
                    X.append(frame)
                    y.append(1)  # 1 = AI-generated
                ai_count += 1
                if ai_count % 10 == 0:
                    print(f"  Processed {ai_count} AI videos...")
    else:
        print(f"Warning: AI videos directory not found: {ai_videos_dir}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to include channel dimension
    if len(X) > 0:
        X = np.expand_dims(X, axis=-1)
    
    print(f"\nDataset Summary:")
    print(f"  Total frames: {len(X)}")
    print(f"  Real frames: {np.sum(y == 0)}")
    print(f"  AI frames: {np.sum(y == 1)}")
    print(f"  Real videos processed: {real_count}")
    print(f"  AI videos processed: {ai_count}")
    
    return X, y


# -----------------------------------------
# 3. Data Augmentation
# -----------------------------------------
def augment_frame(frame):
    """Apply random augmentation to a frame."""
    # Random brightness adjustment
    if np.random.random() > 0.5:
        brightness = np.random.uniform(0.8, 1.2)
        frame = np.clip(frame * brightness, 0, 1)
    
    # Random contrast adjustment
    if np.random.random() > 0.5:
        contrast = np.random.uniform(0.8, 1.2)
        frame = np.clip((frame - 0.5) * contrast + 0.5, 0, 1)
    
    # Random rotation (small angles)
    if np.random.random() > 0.5:
        angle = np.random.uniform(-5, 5)
        center = (64, 64)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        frame = cv2.warpAffine(frame, M, (128, 128))
    
    return frame


def augment_dataset(X, y, augmentation_factor=2):
    """Augment the dataset by creating modified versions of existing frames."""
    X_aug = list(X)
    y_aug = list(y)
    
    print(f"\nAugmenting dataset (factor: {augmentation_factor})...")
    original_size = len(X)
    
    for _ in range(augmentation_factor):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        for idx in indices:
            augmented_frame = augment_frame(X[idx].squeeze())
            X_aug.append(np.expand_dims(augmented_frame, axis=-1))
            y_aug.append(y[idx])
    
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    
    print(f"  Original size: {original_size}")
    print(f"  Augmented size: {len(X_aug)}")
    
    return X_aug, y_aug


# -----------------------------------------
# 4. Training Function
# -----------------------------------------
def train_model(X, y, model_save_path='models/artifact_detector.h5', 
                validation_split=0.2, epochs=50, batch_size=32, 
                use_augmentation=True):
    """
    Train the CNN model.
    
    Args:
        X: Training data (frames)
        y: Training labels
        model_save_path: Where to save the trained model
        validation_split: Fraction of data to use for validation
        epochs: Number of training epochs
        batch_size: Batch size for training
        use_augmentation: Whether to use data augmentation
    """
    if len(X) == 0:
        raise ValueError("No training data available!")
    
    # Create models directory
    os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.', exist_ok=True)
    
    # Split data
    print("\nSplitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42, stratify=y
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Augment training data if requested
    if use_augmentation:
        X_train, y_train = augment_dataset(X_train, y_train, augmentation_factor=2)
    
    # Build model
    print("\nBuilding model...")
    model = build_tf_model()
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on validation set
    print("\nEvaluating model...")
    val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Validation Precision: {val_precision:.4f}")
    print(f"  Validation Recall: {val_recall:.4f}")
    
    # Predictions for detailed metrics
    y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Real', 'AI-Generated']))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    
    # Save training history
    history_path = model_save_path.replace('.h5', '_history.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in values] for k, values in history.history.items()}, f)
    print(f"\nTraining history saved to: {history_path}")
    
    print(f"\n✓ Model saved to: {model_save_path}")
    return model, history


# -----------------------------------------
# 5. Main Function
# -----------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train CNN model for video artifact detection')
    parser.add_argument('--real-dir', type=str, default=r'C:\Users\kshau\Downloads\archive\real',
                        help='Directory containing real video files')
    parser.add_argument('--ai-dir', type=str, default=r'C:\Users\kshau\Downloads\archive\ai',
                        help='Directory containing AI-generated video files')
    parser.add_argument('--output', type=str, default='models/artifact_detector.h5',
                        help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--max-frames', type=int, default=10,
                        help='Maximum frames to extract per video')
    parser.add_argument('--sample-every', type=int, default=30,
                        help='Sample every Nth frame')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable data augmentation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AEGIS CNN Model Training")
    print("=" * 60)
    
    # Load dataset
    X, y = load_dataset(
        args.real_dir,
        args.ai_dir,
        max_frames_per_video=args.max_frames,
        sample_every_n=args.sample_every
    )
    
    if len(X) == 0:
        print("\n❌ Error: No data loaded. Please check your dataset directories.")
        print(f"   Real videos directory: {args.real_dir}")
        print(f"   AI videos directory: {args.ai_dir}")
        return
    
    # Train model
    model, history = train_model(
        X, y,
        model_save_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_augmentation=not args.no_augmentation
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()

