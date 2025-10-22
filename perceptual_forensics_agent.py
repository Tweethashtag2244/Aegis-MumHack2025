# =========================================
# AEGIS Perceptual Forensics (Local Flask Version)
# =========================================

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import json

# -----------------------------------------
# 1. Initialize Flask app
# -----------------------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------------------
# 2. Helper Functions
# -----------------------------------------
def compute_spectral_slope(frame):
    """Compute 1/f spectral slope for a single frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    s = min(h, w)
    crop = gray[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]

    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ps = np.abs(fshift)**2

    center = s // 2
    radial = []
    y, x = np.ogrid[:s, :s]
    for r in range(1, center):
        mask = ((y-center)**2 + (x-center)**2 >= (r-0.5)**2) & ((y-center)**2 + (x-center)**2 < (r+0.5)**2)
        vals = ps[mask]
        if vals.size:
            radial.append(np.mean(vals))
    radial = np.array(radial)
    if radial.size < 3:
        return 0.0

    freqs = np.arange(1, radial.size + 1)
    logf = np.log(freqs)
    logp = np.log(radial + 1e-12)
    slope, _ = np.polyfit(logf, logp, 1)
    return float(slope)


def compute_noise_residual(frame):
    """Estimate the noise residual variance of a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray - blurred
    return float(np.var(residual))


def build_tf_model():
    """Build a small CNN for artifact probability estimation."""
    model = models.Sequential([
        layers.Input(shape=(128, 128, 1)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def detect_fake_probability(frame, model):
    """Use TensorFlow model to infer fake probability."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128)).astype('float32') / 255.0
    inp = np.expand_dims(np.expand_dims(resized, axis=0), axis=-1)
    prob = float(model.predict(inp, verbose=0)[0][0])
    return prob


def save_anomaly_frame(frame, frame_id, outdir="frames"):
    os.makedirs(outdir, exist_ok=True)
    fname = f"{outdir}/frame_{frame_id:03d}.png"
    cv2.imwrite(fname, frame)
    return fname


# -----------------------------------------
# 3. Main analysis logic
# -----------------------------------------
def run_perceptual_forensics(video_path):
    BASELINE_SLOPE = -3.0
    AI_SLOPE_THRESHOLD = -3.4
    tf_model = build_tf_model()

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    sample_every_n = 5

    spectral_slopes = []
    noise_residuals = []
    fake_probs = []
    anomaly_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % sample_every_n != 0:
            continue

        slope = compute_spectral_slope(frame)
        residual = compute_noise_residual(frame)
        prob = detect_fake_probability(frame, tf_model)

        spectral_slopes.append(slope)
        noise_residuals.append(residual)
        fake_probs.append(prob)

        if slope < AI_SLOPE_THRESHOLD or residual < 1000:
            anomaly_frames.append(save_anomaly_frame(frame, frame_count))

    cap.release()

    avg_slope = float(np.mean(spectral_slopes))
    slope_dev = abs(avg_slope - BASELINE_SLOPE)
    avg_residual = float(np.mean(noise_residuals))
    avg_prob = float(np.mean(fake_probs))
    avg_fft_dev = float(np.mean([abs(s - BASELINE_SLOPE) * 1000 for s in spectral_slopes]))

    spectral_weight = 0.7
    noise_weight = 0.2
    cnn_weight = 0.1

    slope_term = min(slope_dev / 0.5, 1.0)
    noise_term = max(0.0, min((10000 - avg_residual) / 10000, 1.0))
    cnn_term = avg_prob

    perceptual_score = (1 - (spectral_weight * slope_term + noise_weight * noise_term + cnn_weight * cnn_term))
    perceptual_score = round(max(0.0, min(perceptual_score, 1.0)), 4)

    explanation = (
        f"The video’s spectral slope is {avg_slope:.2f}, compared to the baseline mean of {BASELINE_SLOPE:.2f}. "
        f"Slope deviation = {slope_dev:.2f}. Noise variance = {avg_residual:.2f}. "
        f"AI artifact probability (TensorFlow): {avg_prob:.2f}."
    )

    result = {
        "perceptualScore": perceptual_score,
        "explanation": explanation,
        "forensicIndicators": {
            "avgFFTDeviation": avg_fft_dev,
            "avgNoiseResidual": avg_residual,
            "modelConfidence": avg_prob,
            "spectralSlope": avg_slope
        },
        "anomalyFrames": anomaly_frames
    }

    return result


# -----------------------------------------
# 4. Flask Endpoint
# -----------------------------------------
@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video_file.save(tmp.name)
        video_path = tmp.name

    try:
        result = run_perceptual_forensics(video_path)
        return jsonify(result)
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


# -----------------------------------------
# 5. Run Server
# -----------------------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
