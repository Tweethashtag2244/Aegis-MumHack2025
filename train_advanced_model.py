import os
import cv2
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Standard size for MobileNetV2
IMG_SIZE = 224 

def extract_frames(video_path, max_frames=20, sample_every=10):
    """Extracts frames, keeping color (RGB) and resizing to 224x224."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    count = 0
    saved_count = 0
    
    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % sample_every == 0:
            # Resize to 224x224
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            # Convert BGR (OpenCV standard) to RGB (TensorFlow standard)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Normalize to [-1, 1] for MobileNetV2
            frame = frame.astype(np.float32)
            frame = (frame / 127.5) - 1.0
            frames.append(frame)
            saved_count += 1
        count += 1
        
    cap.release()
    return np.array(frames)

def load_dataset(real_dir, ai_dir):
    X = []
    y = []
    
    print(f"Loading Real videos from {real_dir}...")
    for filename in os.listdir(real_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            path = os.path.join(real_dir, filename)
            frames = extract_frames(path)
            if len(frames) > 0:
                X.append(frames)
                y.extend([0] * len(frames)) # 0 = Real
                
    print(f"Loading AI videos from {ai_dir}...")
    for filename in os.listdir(ai_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            path = os.path.join(ai_dir, filename)
            frames = extract_frames(path)
            if len(frames) > 0:
                X.append(frames)
                y.extend([1] * len(frames)) # 1 = AI

    if len(X) == 0:
        return np.array([]), np.array([])
        
    X = np.concatenate(X)
    y = np.array(y)
    return X, y

def build_advanced_model():
    """Builds a model using MobileNetV2 as the base."""
    # Load MobileNetV2 without the top layer, pre-trained on ImageNet
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Freeze the base model layers (so we don't destroy pre-trained patterns)
    base_model.trainable = False 
    
    # Add new classification layers
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)  # Dropout to prevent overfitting
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Update defaults to your actual paths
    parser.add_argument('--real-dir', type=str, default=r"C:\Users\kshau\Downloads\archive\real")
    parser.add_argument('--ai-dir', type=str, default=r"C:\Users\kshau\Downloads\archive\ai")
    parser.add_argument('--output', type=str, default='models/artifact_detector.h5')
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    # 1. Load Data
    print("--- Preparing Data ---")
    X, y = load_dataset(args.real_dir, args.ai_dir)
    print(f"Total frames: {len(X)}")
    
    if len(X) == 0:
        print("Error: No data found. Check your paths.")
        exit()

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 2. Build Model (Transfer Learning)
    print("--- Building MobileNetV2 Model ---")
    model = build_advanced_model()
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 3. Train
    print("--- Starting Training ---")
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint(args.output, save_best_only=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.2, patience=3)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=32,
        callbacks=callbacks
    )
    
    print(f"--- Training Complete. Model saved to {args.output} ---")