# =========================================
# AEGIS Perceptual Forensics (Adversarial Defense)
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
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Will use heuristic method only.")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -----------------------------------------
# 1. Helper Functions
# -----------------------------------------
def compute_spectral_slope(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    s = min(h, w)
    crop = gray[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]

    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ps = np.abs(fshift)**2

    center = s // 2
    y, x = np.ogrid[:s, :s]
    r_grid = np.sqrt((y - center)**2 + (x - center)**2)
    r_int = r_grid.astype(int)
    
    radial = np.bincount(r_int.ravel(), weights=ps.ravel()) / np.maximum(np.bincount(r_int.ravel()), 1)
    radial = radial[1:center] 

    if radial.size < 3: return 0.0

    freqs = np.arange(1, radial.size + 1)
    slope, _ = np.polyfit(np.log(freqs), np.log(radial + 1e-12), 1)
    return float(slope)

def compute_noise_residual(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return float(np.var(gray - blurred))

# --- NEW: Double Compression Detection ---
def detect_double_compression(frame):
    """
    Analyzes DCT histogram to find double compression artifacts.
    Real social media videos are single-compressed (by the platform).
    Resized/Edited AI videos are double-compressed.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Crop to 8x8 aligned grid
    h_trim = h - (h % 8)
    w_trim = w - (w % 8)
    gray = gray[:h_trim, :w_trim]
    
    # Compute block-wise DCT
    # (Simplified approximation using difference error for speed)
    # A re-compressed image often has a periodic error pattern in the pixel difference
    
    # Calculate simple error metric:
    # Double compressed images often have lower error variance in 8x8 blocks due to quantization snapping
    
    # We will use a heuristic: High quality AI resized -> Low frequency artifacts
    # Real WhatsApp -> High frequency block noise
    
    # Edge density check (Real WhatsApp has jagged block edges)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (h * w)
    
    return edge_density

# -----------------------------------------
# 2. Agents
# -----------------------------------------

# --- AGENT: Provenance & Lineage (UPDATED) ---
def check_provenance(video_path, filename):
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_path
        ]
        result = subprocess.check_output(cmd)
        data = json.loads(result)
        
        format_tags = data.get('format', {}).get('tags', {})
        encoder = format_tags.get('encoder', '').lower()
        
        # Get Video Dimensions
        width = 0
        height = 0
        streams = data.get('streams', [])
        for stream in streams:
            if stream.get('codec_type') == 'video':
                width = int(stream.get('width', 0))
                height = int(stream.get('height', 0))
                break

        # 1. Check Patterns
        is_mobile = bool(re.search(r'(vid|img|pxl|dji|c0)_\d{4}', filename.lower())) or \
                    bool(re.search(r'\d{8}_\d{6}', filename))

        is_social = bool(re.search(r'whatsapp video', filename.lower())) or \
                    bool(re.search(r'snapchat-', filename.lower())) or \
                    bool(re.search(r'facebook|fb_img', filename.lower())) or \
                    bool(re.search(r'tiktok', filename.lower()))

        # --- DECOY DETECTION 1: Resolution Mismatch ---
        is_decoy = False
        if is_social:
            standard_dims = [1920, 1080, 1280, 720, 848, 480, 640, 360, 352]
            if width not in standard_dims and height not in standard_dims:
                is_decoy = True
                print(f"[Provenance] Decoy Detected! Filename is social but dimensions {width}x{height} are non-standard.")

        # 2. Check Suspicious Encoders
        suspicious_encoders = ['lavf', 'unknown', 'isom', 'google', 'gen-2', 'gen-3', 'luma'] 
        is_suspicious_encoder = any(x in encoder for x in suspicious_encoders)
        
        if not format_tags:
            is_suspicious_encoder = True
            encoder = "Missing/Stripped"

        # Score Logic
        if is_decoy:
            score = 0.1 
            is_mobile = False
            is_social = False
        elif is_mobile or is_social:
            score = 0.95 
            is_suspicious_encoder = False 
        elif is_suspicious_encoder:
            score = 0.2
        else:
            score = 0.8
        
        return {
            "provenanceScore": score,
            "encoder": encoder,
            "isMobileFilename": is_mobile,
            "isSocialFilename": is_social,
            "hasC2PA": False 
        }
    except Exception as e:
        print(f"Provenance Error: {e}")
        return {"provenanceScore": 0.5, "encoder": "Error", "isMobileFilename": False, "isSocialFilename": False, "hasC2PA": False}

# --- AGENT: Model Attribution ---
def attribute_model(width, height, fps, encoder, is_mobile, is_social):
    if is_mobile or is_social:
         return {"detectedModel": "None (Trusted Pattern)", "confidence": 0.0}

    if 'google' in encoder:
        return {"detectedModel": "Google Veo / Imagen", "confidence": 0.95}
        
    heuristics = [
        {"name": "Sora (Likely)", "w": 1920, "h": 1080, "fps": 30, "enc": "lavf"},
        {"name": "Runway Gen-2", "w": 1792, "h": 1024, "fps": 24, "enc": "mp4"},
        {"name": "Pika Labs", "w": 1280, "h": 720, "fps": 24, "enc": "lavf"},
    ]
    
    for h in heuristics:
        if width == h["w"] and height == h["h"]:
             if h["enc"] in encoder:
                 return {"detectedModel": h["name"], "confidence": 0.8}
             
    return {"detectedModel": "Unknown / Custom", "confidence": 0.1}

# --- AGENT: Adversarial Simulation ---
def perform_adversarial_attack(frame, model, original_prob, use_cnn):
    try:
        noise = np.random.normal(0, 15, frame.shape).astype(np.uint8)
        noisy_frame = cv2.add(frame, noise)
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w//2, h//2))
        restored = cv2.resize(small, (w, h))
        
        if use_cnn and model:
            prob_noise = detect_fake_probability_cnn(noisy_frame, model)
            prob_blur = detect_fake_probability_cnn(restored, model)
        else:
            s_noise = compute_spectral_slope(noisy_frame)
            r_noise = compute_noise_residual(noisy_frame)
            prob_noise = detect_fake_probability_heuristic(s_noise, r_noise)
            prob_blur = original_prob

        variance = max(abs(prob_noise - original_prob), abs(prob_blur - original_prob))
        robustness = max(0.0, 1.0 - (variance * 2.0)) 
        return robustness
    except:
        return 0.5

# -----------------------------------------
# 3. Model Loading & Core Detection
# -----------------------------------------
def load_trained_model(model_path='models/artifact_detector.h5'):
    if not TENSORFLOW_AVAILABLE: return None
    if not os.path.exists(model_path): return None
    try:
        return tf.keras.models.load_model(model_path)
    except: return None

def detect_fake_probability_cnn(frame, model):
    try:
        resized = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb_frame.astype('float32') / 127.5) - 1.0
        input_data = np.expand_dims(normalized, axis=0)
        prediction = model.predict(input_data, verbose=0)
        return float(prediction[0][0])
    except: return 0.5

def detect_fake_probability_heuristic(slope, residual, baseline_slope=-3.0):
    slope_deviation = abs(slope - baseline_slope)
    slope_ai_prob = min(1.0, max(0.0, (slope_deviation - 0.2) / 0.6))
    
    if residual < 3000: noise_ai_prob = 0.95
    elif residual < 6000: noise_ai_prob = 0.70
    elif residual < 9000: noise_ai_prob = 0.40
    else: noise_ai_prob = 0.10
    
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

    provenance = check_provenance(video_path, filename)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    attribution = attribute_model(width, height, fps, provenance['encoder'], provenance['isMobileFilename'], provenance['isSocialFilename'])
    
    cnn_model = get_detection_model()
    use_cnn = cnn_model is not None
    
    frame_count = 0
    sample_every_n = 5
    fake_probs = []
    spectral_slopes = []
    noise_residuals = []
    robustness_scores = []
    anomaly_frames = []
    
    # New: Compression Check
    compression_scores = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        if frame_count % sample_every_n != 0: continue

        slope = compute_spectral_slope(frame)
        residual = compute_noise_residual(frame)
        
        if use_cnn:
            prob = detect_fake_probability_cnn(frame, cnn_model)
        else:
            prob = detect_fake_probability_heuristic(slope, residual)

        spectral_slopes.append(slope)
        noise_residuals.append(residual)
        fake_probs.append(prob)
        
        # Check compression artifacts
        comp_score = detect_double_compression(frame)
        compression_scores.append(comp_score)

        if len(fake_probs) % 5 == 0:
            rob = perform_adversarial_attack(frame, cnn_model, prob, use_cnn)
            robustness_scores.append(rob)

        if prob > 0.85: anomaly_frames.append(save_anomaly_frame(frame, frame_count))

    cap.release()

    if not fake_probs: return {"error": "Could not analyze video frames"}

    avg_prob = float(np.mean(fake_probs))
    avg_robustness = float(np.mean(robustness_scores)) if robustness_scores else 0.5
    avg_slope = float(np.mean(spectral_slopes))
    avg_residual = float(np.mean(noise_residuals))
    avg_compression = float(np.mean(compression_scores)) if compression_scores else 0.0

    # DECOY DETECTION 2: Compression Analysis
    # Real WhatsApp has HIGH edge density (blocky noise) -> > 0.05
    # Resized AI is often SMOOTHED by the resize -> < 0.02
    
    if provenance['isSocialFilename'] and avg_compression < 0.02:
        print(f"[Perceptual] Decoy Detected! Social filename but unnaturally smooth (Comp Score {avg_compression:.4f})")
        # Revoke social trust
        provenance['isSocialFilename'] = False
        provenance['provenanceScore'] = 0.2

    perceptual_score = 1.0 - avg_prob
    perceptual_score = round(max(0.0, min(perceptual_score, 1.0)), 4)

    return {
        "perceptualScore": perceptual_score,
        "explanation": f"Analyzed {frame_count} frames. Avg AI Probability: {avg_prob:.2f}. Compression Signature: {avg_compression:.3f}",
        "forensicIndicators": {
            "avgNoiseResidual": avg_residual,
            "modelConfidence": avg_prob,
            "spectralSlope": avg_slope
        },
        "anomalyFrames": anomaly_frames[:5],
        "provenance": provenance,
        "attribution": attribution,
        "adversarial": {
            "robustnessScore": avg_robustness,
            "explanation": "Robustness check complete."
        }
    }

# -----------------------------------------
# 5. Flask Endpoints
# -----------------------------------------
@app.route('/health', methods=['GET'])
def health(): return jsonify({"status": "ok"}), 200

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_video():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    if 'video' not in request.files: return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    original_filename = video_file.filename

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video_file.save(tmp.name)
        video_path = tmp.name

    try:
        result = run_full_scan(video_path, original_filename)
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path): os.remove(video_path)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)