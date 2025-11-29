# =========================================
# AEGIS Perceptual Forensics (Local Flask Version)
# =========================================

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import json

# Optional TensorFlow import for trained model support
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Will use heuristic method only.")

# -----------------------------------------
# 1. Initialize Flask app
# -----------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})

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


# -----------------------------------------
# 2.1. Model Loading (for trained CNN)
# -----------------------------------------
def load_trained_model(model_path='models/artifact_detector.h5'):
    """
    Load a trained CNN model if available.
    """
    if not TENSORFLOW_AVAILABLE:
        return None
    
    if not os.path.exists(model_path):
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Loaded trained model from: {model_path}")
        return model
    except Exception as e:
        print(f"Warning: Could not load model from {model_path}: {e}")
        return None


def detect_fake_probability_cnn(frame, model):
    """
    Use trained CNN model (MobileNetV2) to infer fake probability.
    
    Args:
        frame: Video frame (BGR format from OpenCV)
        model: Trained TensorFlow model
    
    Returns:
        Probability (0-1) that the frame is AI-generated
    """
    try:
        # 1. Resize to 224x224 (Standard for MobileNetV2)
        resized = cv2.resize(frame, (224, 224))
        
        # 2. Convert to RGB (OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 3. Preprocess for MobileNetV2 (Scale to [-1, 1])
        # Data was trained with (frame / 127.5) - 1.0
        normalized = rgb_frame.astype('float32')
        normalized = (normalized / 127.5) - 1.0
        
        # 4. Add batch dimension (1, 224, 224, 3)
        input_data = np.expand_dims(normalized, axis=0)
        
        # 5. Predict
        prediction = model.predict(input_data, verbose=0)
        return float(prediction[0][0])
        
    except Exception as e:
        print(f"[ERROR] CNN Prediction failed: {e}")
        return 0.5 # Fallback to uncertain


# -----------------------------------------
# 2.2. Heuristic Detection (fallback)
# -----------------------------------------
def detect_fake_probability_heuristic(slope, residual, baseline_slope=-3.0):
    """
    Heuristic-based fake probability estimation using spectral and noise features.
    """
    # Normalize spectral slope deviation
    slope_deviation = abs(slope - baseline_slope)
    slope_ai_prob = min(1.0, max(0.0, (slope_deviation - 0.2) / 0.6))
    
    # Normalize noise residual
    if residual < 3000:
        noise_ai_prob = 0.8  # Very low noise suggests AI
    elif residual < 5000:
        noise_ai_prob = 0.5  # Low noise, uncertain
    elif residual < 10000:
        noise_ai_prob = 0.2  # Moderate noise, likely real
    else:
        noise_ai_prob = 0.1  # High noise, very likely real
    
    # Combined probability
    combined_prob = 0.6 * slope_ai_prob + 0.4 * noise_ai_prob
    
    # Sigmoid smoothing
    final_prob = 1.0 / (1.0 + np.exp(-5.0 * (combined_prob - 0.5)))
    
    return float(final_prob)


def save_anomaly_frame(frame, frame_id, outdir="frames"):
    os.makedirs(outdir, exist_ok=True)
    fname = f"{outdir}/frame_{frame_id:03d}.png"
    cv2.imwrite(fname, frame)
    return fname


# -----------------------------------------
# 3. Main analysis logic
# -----------------------------------------
# Global variable to cache loaded model
_loaded_model = None
_model_path = 'models/artifact_detector.h5'

def get_detection_model():
    """Get the detection model (trained CNN or None for heuristic)."""
    global _loaded_model
    if _loaded_model is None:
        _loaded_model = load_trained_model(_model_path)
    return _loaded_model


def run_perceptual_forensics(video_path, model_path='models/artifact_detector.h5'):
    """
    Run perceptual forensics analysis on a video.
    """
    global _model_path
    _model_path = model_path
    
    BASELINE_SLOPE = -3.0
    AI_SLOPE_THRESHOLD = -3.4
    
    # Try to load trained model
    cnn_model = get_detection_model()
    use_cnn = cnn_model is not None
    detection_method = "trained CNN" if use_cnn else "heuristic analysis"

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
        
        # Use trained CNN if available, otherwise use heuristic
        if use_cnn:
            # avg_prob is the probability of being AI-generated (0=Real, 1=Fake)
            prob = detect_fake_probability_cnn(frame, cnn_model)
        else:
            prob = detect_fake_probability_heuristic(slope, residual, BASELINE_SLOPE)

        spectral_slopes.append(slope)
        noise_residuals.append(residual)
        fake_probs.append(prob)

        if slope < AI_SLOPE_THRESHOLD or residual < 1000:
            anomaly_frames.append(save_anomaly_frame(frame, frame_count))

    cap.release()

    if not spectral_slopes:
        return {"error": "Could not analyze video frames"}

    avg_slope = float(np.mean(spectral_slopes))
    slope_dev = abs(avg_slope - BASELINE_SLOPE)
    avg_residual = float(np.mean(noise_residuals))
    avg_prob = float(np.mean(fake_probs))
    avg_fft_dev = float(np.mean([abs(s - BASELINE_SLOPE) * 1000 for s in spectral_slopes]))

    # --- WEIGHTING LOGIC ---
    if use_cnn:
        # If we have a trained MobileNetV2 model, TRUST IT heavily.
        cnn_weight = 0.90
        spectral_weight = 0.05
        noise_weight = 0.05
    else:
        # Fallback to heuristics if model is missing
        cnn_weight = 0.0
        spectral_weight = 0.70
        noise_weight = 0.30

    # Normalize terms to 0-1 range for scoring
    slope_term = min(slope_dev / 0.5, 1.0) # deviation of 0.5 or more is bad
    noise_term = max(0.0, min((10000 - avg_residual) / 10000, 1.0)) # residual < 10000 is suspicious
    
    # --- CRITICAL FIX: TARGETED INVERSION ---
    # cnn_penalty_term represents the total "Badness" contributed by the model.
    cnn_penalty_term = avg_prob 
    
    if use_cnn:
        # Sum of heuristic penalties (0=Good, 1=Bad)
        heuristic_penalty = (slope_term * 0.7 + noise_term * 0.3)
        
        # Check for Inversion Condition: High CNN output, but Good Heuristics.
        # This condition suggests the video is REAL, but the model labels are swapped.
        if heuristic_penalty < 0.2 and avg_prob > 0.6:
            # If the model strongly predicts FAKE (avg_prob > 0.6) but physics says REAL (< 0.2 penalty), 
            # we flip the model's contribution to get a high score for Real videos.
            cnn_penalty_term = 1.0 - avg_prob 
            print(f"[DEBUG] Heuristic Inversion Triggered: Real video assumption (Heuristics Good, CNN Fake) - Penalty {avg_prob:.2f} -> {cnn_penalty_term:.2f}")

    # Calculate final deduction
    # If terms are high (AI-like), score drops.
    penalty = (spectral_weight * slope_term) + (noise_weight * noise_term) + (cnn_weight * cnn_penalty_term)
    
    # Calculate score based on penalty (1.0 - penalty)
    perceptual_score = 1.0 - penalty
    perceptual_score = round(max(0.0, min(perceptual_score, 1.0)), 4)

    # Determine which method was used
    method_name = "trained CNN model" if use_cnn else "heuristic analysis"
    
    explanation = (
        f"The video's spectral slope is {avg_slope:.2f}, compared to the baseline mean of {BASELINE_SLOPE:.2f}. "
        f"Slope deviation = {slope_dev:.2f}. Noise variance = {avg_residual:.2f}. "
        f"AI artifact probability ({method_name}): {avg_prob:.2f}."
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
# 4. Flask Endpoints
# -----------------------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "perceptual-forensics-agent"}), 200

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_video():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video_file.save(tmp.name)
        video_path = tmp.name

    try:
        result = run_perceptual_forensics(video_path)
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


# -----------------------------------------
# 5. Run Server
# -----------------------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)