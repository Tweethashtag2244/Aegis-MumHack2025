# =========================================
# AEGIS Perceptual Forensics (Full Multi-Agent Version)
# =========================================

import os
import cv2
import numpy as np
import subprocess
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile

# Optional TensorFlow import
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Will use heuristic method only.")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -----------------------------------------
# 1. Helper Functions (Perceptual Agent)
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
    # Optimized radial profile calculation
    y, x = np.ogrid[:s, :s]
    r_grid = np.sqrt((y - center)**2 + (x - center)**2)
    r_int = r_grid.astype(int)
    
    # Vectorized radial mean using bincount
    radial = np.bincount(r_int.ravel(), weights=ps.ravel()) / np.maximum(np.bincount(r_int.ravel()), 1)
    radial = radial[1:center] # Skip DC and edges

    if radial.size < 3: return 0.0

    freqs = np.arange(1, radial.size + 1)
    slope, _ = np.polyfit(np.log(freqs), np.log(radial + 1e-12), 1)
    return float(slope)

def compute_noise_residual(frame):
    """Estimate the noise residual variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return float(np.var(gray - blurred))

# -----------------------------------------
# 2. New Agents: Provenance, Attribution, Adversarial
# -----------------------------------------

# --- AGENT: Provenance & Lineage ---
def check_provenance(video_path, filename):
    """Extracts metadata to find editing signatures or generative tags."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_path
        ]
        # Use subprocess to call ffprobe
        result = subprocess.check_output(cmd)
        data = json.loads(result)
        
        format_tags = data.get('format', {}).get('tags', {})
        encoder = format_tags.get('encoder', '').lower()
        creation_time = format_tags.get('creation_time', None)
        
        # 1. Check Filename Patterns (New)
        # Real phones use patterns like "VID_2025..." or "IMG_..." or "20250903..."
        is_mobile_filename = bool(re.search(r'(vid|img|pxl|dji)_\d{8}', filename.lower())) or \
                             bool(re.search(r'\d{8}_\d{6}', filename))

        # 2. Check Suspicious Encoders
        # 'ffmpeg' is common in edits, so it is less suspicious than 'lavf' or 'unknown'
        suspicious_encoders = ['lavf', 'unknown', 'isom'] 
        is_suspicious_encoder = any(x in encoder for x in suspicious_encoders)
        
        # If metadata is completely empty, that is also suspicious
        if not format_tags:
            is_suspicious_encoder = True
            encoder = "Missing/Stripped"

        # Score Logic
        if is_mobile_filename:
            score = 0.95 # Highly likely real if filename matches camera pattern
        elif is_suspicious_encoder:
            score = 0.3 
        else:
            score = 0.8
        
        return {
            "provenanceScore": score,
            "encoder": encoder,
            "creationTime": creation_time,
            "isMobileFilename": is_mobile_filename,
            "hasC2PA": False # Placeholder: Real C2PA requires a dedicated library
        }
    except Exception as e:
        print(f"Provenance Error: {e}")
        return {"provenanceScore": 0.5, "encoder": "Error", "isMobileFilename": False, "hasC2PA": False}

# --- AGENT: Model Attribution ---
def attribute_model(width, height, fps, encoder, is_mobile_filename):
    """Guesses the AI generator based on resolution and encoding fingerprints."""
    
    # If it looks like a phone file, DO NOT guess it's a model
    if is_mobile_filename:
         return {"detectedModel": "None (Mobile Pattern)", "confidence": 0.0}

    # Common fingerprints for current gen AI video tools
    heuristics = [
        {"name": "Sora (Likely)", "w": 1920, "h": 1080, "fps": 30, "enc": "lavf"},
        {"name": "Runway Gen-2", "w": 1792, "h": 1024, "fps": 24, "enc": "mp4"},
        {"name": "Pika Labs", "w": 1280, "h": 720, "fps": 24, "enc": "lavf"},
    ]
    
    for h in heuristics:
        # Simple fuzzy match on resolution
        if width == h["w"] and height == h["h"]:
             # Stricter check: Encoder must also match partially if defined in heuristic
             if h["enc"] in encoder:
                 return {"detectedModel": h["name"], "confidence": 0.8}
             elif encoder == "unknown" or encoder == "":
                 # If encoder is hidden, it's a weak "Maybe"
                 return {"detectedModel": h["name"] + "?", "confidence": 0.4}
             
    return {"detectedModel": "Unknown / Custom", "confidence": 0.1}

# --- AGENT: Adversarial Simulation ---
def perform_adversarial_attack(frame, model, original_prob, use_cnn):
    """Applies noise/blur to see if the detection is fragile."""
    try:
        # Attack 1: Gaussian Noise (Simulates compression grain)
        noise = np.random.normal(0, 15, frame.shape).astype(np.uint8)
        noisy_frame = cv2.add(frame, noise)
        
        # Attack 2: Downscale-Upscale (Simulates social media compression)
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w//2, h//2))
        restored = cv2.resize(small, (w, h))
        
        if use_cnn and model:
            prob_noise = detect_fake_probability_cnn(noisy_frame, model)
            prob_blur = detect_fake_probability_cnn(restored, model)
        else:
            # Heuristic fallback for adversarial check
            # Real cameras maintain slope relationships; AI often breaks under noise
            s_noise = compute_spectral_slope(noisy_frame)
            r_noise = compute_noise_residual(noisy_frame)
            prob_noise = detect_fake_probability_heuristic(s_noise, r_noise)
            prob_blur = original_prob # Heuristics are less sensitive to blur

        # Logic: If adding simple noise changes the probability by > 30%, the model is "fragile"
        variance = max(abs(prob_noise - original_prob), abs(prob_blur - original_prob))
        
        # Robustness Score: 1.0 (Very Robust) -> 0.0 (Very Fragile)
        robustness = max(0.0, 1.0 - (variance * 2.0)) 
        
        return robustness
    except Exception as e:
        print(f"Adversarial Error: {e}")
        return 0.5

# -----------------------------------------
# 3. Model Loading & Core Detection
# -----------------------------------------
def load_trained_model(model_path='models/artifact_detector.h5'):
    if not TENSORFLOW_AVAILABLE: return None
    if not os.path.exists(model_path): return None
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Loaded trained model from: {model_path}")
        return model
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        return None

def detect_fake_probability_cnn(frame, model):
    try:
        resized = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb_frame.astype('float32') / 127.5) - 1.0
        input_data = np.expand_dims(normalized, axis=0)
        prediction = model.predict(input_data, verbose=0)
        return float(prediction[0][0])
    except:
        return 0.5

def detect_fake_probability_heuristic(slope, residual, baseline_slope=-3.0):
    slope_deviation = abs(slope - baseline_slope)
    slope_ai_prob = min(1.0, max(0.0, (slope_deviation - 0.2) / 0.6))
    
    if residual < 3000: noise_ai_prob = 0.8
    elif residual < 5000: noise_ai_prob = 0.5
    elif residual < 10000: noise_ai_prob = 0.2
    else: noise_ai_prob = 0.1
    
    combined = 0.6 * slope_ai_prob + 0.4 * noise_ai_prob
    return 1.0 / (1.0 + np.exp(-5.0 * (combined - 0.5)))

def save_anomaly_frame(frame, frame_id, outdir="frames"):
    os.makedirs(outdir, exist_ok=True)
    fname = f"{outdir}/frame_{frame_id:03d}.png"
    cv2.imwrite(fname, frame)
    return fname

# -----------------------------------------
# 4. Main Execution Logic
# -----------------------------------------
_loaded_model = None
_model_path = 'models/artifact_detector.h5'

def get_detection_model():
    global _loaded_model
    if _loaded_model is None:
        _loaded_model = load_trained_model(_model_path)
    return _loaded_model

def run_full_scan(video_path, filename="unknown_file", model_path='models/artifact_detector.h5'):
    global _model_path
    _model_path = model_path
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}

    # --- 1. Run Provenance Agent (Metadata) ---
    provenance = check_provenance(video_path, filename)
    
    # --- 2. Run Model Attribution Agent (Fingerprints) ---
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    attribution = attribute_model(width, height, fps, provenance['encoder'], provenance['isMobileFilename'])
    
    # --- 3. Run Perceptual & Adversarial Agents ---
    cnn_model = get_detection_model()
    use_cnn = cnn_model is not None
    
    frame_count = 0
    sample_every_n = 5
    
    spectral_slopes = []
    noise_residuals = []
    fake_probs = []
    robustness_scores = []
    anomaly_frames = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        if frame_count % sample_every_n != 0: continue

        # Perceptual Checks
        slope = compute_spectral_slope(frame)
        residual = compute_noise_residual(frame)
        
        if use_cnn:
            prob = detect_fake_probability_cnn(frame, cnn_model)
        else:
            prob = detect_fake_probability_heuristic(slope, residual)

        spectral_slopes.append(slope)
        noise_residuals.append(residual)
        fake_probs.append(prob)

        # Adversarial Checks (Run less frequently for performance)
        if len(fake_probs) % 5 == 0:
            rob = perform_adversarial_attack(frame, cnn_model, prob, use_cnn)
            robustness_scores.append(rob)

        # Save Anomalies
        if prob > 0.85: # High confidence fake frames
            anomaly_frames.append(save_anomaly_frame(frame, frame_count))

    cap.release()

    if not fake_probs:
        return {"error": "Could not analyze video frames"}

    avg_prob = float(np.mean(fake_probs))
    avg_robustness = float(np.mean(robustness_scores)) if robustness_scores else 0.5
    avg_slope = float(np.mean(spectral_slopes))
    avg_residual = float(np.mean(noise_residuals))

    # Construct explanation
    method_name = "trained CNN" if use_cnn else "heuristics"
    explanation = (
        f"Analyzed {frame_count} frames. Avg AI Probability ({method_name}): {avg_prob:.2f}. "
        f"Adversarial Robustness: {avg_robustness:.2f}. "
        f"Provenance: {provenance['encoder']}."
    )

    # Final Perceptual Score (1.0 = Real, 0.0 = Fake)
    perceptual_score = 1.0 - avg_prob
    perceptual_score = round(max(0.0, min(perceptual_score, 1.0)), 4)

    result = {
        "perceptualScore": perceptual_score,
        "explanation": explanation,
        "forensicIndicators": {
            "avgNoiseResidual": avg_residual,
            "modelConfidence": avg_prob,
            "spectralSlope": avg_slope
        },
        "anomalyFrames": anomaly_frames[:5], # Limit return size
        # New Agent Outputs
        "provenance": provenance,
        "attribution": attribution,
        "adversarial": {
            "robustnessScore": avg_robustness,
            "explanation": "High robustness indicates consistent artifacts." if avg_robustness > 0.7 else "Low robustness suggests fragile generation artifacts."
        }
    }

    return result

# -----------------------------------------
# 5. Flask Endpoints
# -----------------------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

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
    original_filename = video_file.filename # Capture the real filename!

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video_file.save(tmp.name)
        video_path = tmp.name

    try:
        # PASS THE FILENAME HERE
        result = run_full_scan(video_path, original_filename)
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)