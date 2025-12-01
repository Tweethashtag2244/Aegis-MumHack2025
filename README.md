# 🛡️ Aegis — Autonomous AI Video Detection & Attribution

Aegis is an **agentic AI framework** that autonomously detects, attributes, and explains **AI-generated or manipulated videos**.  
By combining **signal-level forensics**, **semantic reasoning**, and **provenance verification**, Aegis delivers interpretable, evidence-backed authenticity assessments.

---

## 🚀 Overview

AI-generated videos from tools like **Sora**, **Runway**, **Pika**, and **Synthesia** are becoming increasingly lifelike, blurring the line between real and synthetic media.  
Aegis provides a transparent, explainable verification layer — helping individuals, journalists, and platforms **restore trust in digital content**.

Built as a **multi-agent system**, Aegis evaluates videos through specialized AI agents that analyze different modalities — from pixel noise and lighting to audio semantics and provenance.

---

## 🧠 Core Agents

### 1. Perceptual Forensics Agent

- Extracts frames and analyzes **spectral slope (1/f)**, **noise residual variance**, and **CNN-based artifact detection**.
- Identifies temporal or physical inconsistencies that hint at synthetic generation.
- **Outputs:** `perceptualScore`, `forensicIndicators`, `anomalyFrames`.

### 2. Semantic Coherence Agent

- Uses **Gemini Vision + ASR** to check audio–visual alignment.
- Evaluates **lip-sync**, **speech–scene relevance**, and **background sound realism**.
- **Outputs:** `semanticScore`, `explanation`.

### 3. Model Attribution Agent

- Detects latent **generator fingerprints** to infer the likely AI model (Sora, Runway, Pika, Synthesia).
- Compares residuals with known model signatures.
- **Outputs:** `modelName`, `confidence`.

### 4. Provenance & Lineage Agent

- Extracts EXIF, codec metadata, and timestamps.
- Verifies **C2PA provenance** and blockchain-based signatures.
- Checks earliest web appearance and container history.
- **Outputs:** `provenanceConfidence`, `metadata`.

### 5. Adversarial Simulation Agent

- Generates perturbed versions of the input (recompression, denoising, jitter).
- Measures **robustness and consistency** under perturbations.
- **Outputs:** `robustnessMetrics`, `perturbationBudget`.

### 6. Collective Consensus Agent

- Aggregates verification signals from web sources, reverse search, and trusted databases.
- Computes `consensusScore` and supporting evidence.

---

## 🧩 Tech Stack

- **Languages:** Python, TypeScript
- **Frameworks:** Flask, TensorFlow, OpenCV, Firebase Functions, Vertex AI GenKit
- **Infrastructure:** Docker, Nginx (TLS 1.2), Cloud Run, GCS/S3
- **Monitoring & CI/CD:** Prometheus, Grafana, Sentry, GitHub Actions
- **Models:** Gemini 2.5 Flash (Vision + ASR), Custom CNN (artifact detection)

---

## 🚦 Getting Started & Execution Instructions

### Prerequisites

**System Requirements**

- Windows 10/11, macOS, or Linux
- Python 3.10+
- Node.js 18+ and npm
- FFmpeg binary (for video processing)
- 4GB+ RAM (8GB recommended for model inference)

**Install FFmpeg**

- **Windows (winget):** `winget install --id=Gyan.FFmpeg -e`
- **Windows (Chocolatey):** `choco install ffmpeg -y`
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`
- **Verify:** `ffmpeg -version`

**Environment Variables**
Create a `.env.local` file in the project root with:

```
NEXT_PUBLIC_FIREBASE_API_KEY=<your-firebase-api-key>
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=<your-firebase-auth-domain>
NEXT_PUBLIC_FIREBASE_PROJECT_ID=<your-firebase-project-id>
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=<your-firebase-storage-bucket>
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=<your-sender-id>
NEXT_PUBLIC_FIREBASE_APP_ID=<your-firebase-app-id>
GOOGLE_GENAI_API_KEY=<your-google-genai-api-key>
```

See `TRAINING_INSTRUCTIONS.md` for Firebase and Genkit setup details.

### Installation

**1. Clone & Install Dependencies**

```bash
# Navigate to project directory
cd Aegis-MumHack2025-main

# Install Python dependencies
pip install -r requirements.txt

# Install Node dependencies
npm install
```

**2. Verify Installations**

```bash
# Check Python packages
python -c "import tensorflow, cv2, flask, numpy; print('Python dependencies OK')"

# Check Node packages
npm list genkit firebase react
```

### Running the Application

**Option A: Full Stack (Frontend + Backend + Genkit LLM Flow)**

```bash
# Terminal 1: Start Next.js frontend + dev server
npm run dev:local

# This runs:
# - Next.js on http://localhost:9002
# - Genkit flow watcher for LLM analysis
# - API routes for video upload & analysis

# Then visit: http://localhost:9002
```

**Option B: Frontend Only (Connect to Existing Backend)**

```bash
# Terminal 1: Start Next.js frontend
npm run dev

# Terminal 2: Start Flask backend (if separate deployment)
python -m flask run --port 5000

# Frontend on http://localhost:9002
# Flask API on http://localhost:5000
```

**Option C: Production Build**

```bash
# Build and optimize for production
npm run build

# Start production server
npm start

# Visit: http://localhost:3000
```

### Training & Model Development

**Train CNN Artifact Detector**

```bash
# Windows PowerShell
.\train_model.ps1

# macOS/Linux Bash
bash train_model.ps1

# Or directly with Python
python train_cnn_model.py
```

**Train Advanced Model (MobileNetV2)**

```bash
python train_advanced_model.py
```

**Expected Output**

- Model saved to `models/artifact_detector.h5`
- Training history logged to `models/artifact_detector_history.json`
- Frames extracted to `frames/` directory

### Workflow: Upload & Analyze Video

1. **Open Dashboard:** http://localhost:9002
2. **Upload Video:** Use the video upload form (MP4, WebM, MOV supported)
3. **Processing:**
   - Frontend sends video to `/api/analyze` endpoint
   - Flask + TensorFlow extracts frames and runs forensic analysis
   - Gemini Vision API evaluates semantic coherence
   - Genkit Reasoning Agent synthesizes final verdict
4. **View Results:**
   - Authenticity score (0–1, lower = likely AI-generated)
   - Detailed forensic breakdown (spectral, temporal, semantic signals)
   - Classification (Likely Real / Uncertain / Likely AI-Generated)
   - Narrative reasoning from LLM

### Troubleshooting

**Issue: "Detailed summary unavailable due to API timeout"**

- Increase Genkit request timeout in `src/ai/genkit.ts`
- Reduce `maxOutputTokens` in `src/ai/flows/generate-reasoning-explanation.ts` (currently 1024)
- Ensure `GOOGLE_GENAI_API_KEY` is valid and has quota
- Check network connectivity to Google API endpoints

**Issue: FFmpeg not found**

- Verify installation: `ffmpeg -version`
- Add FFmpeg to system PATH if needed
- For Windows: restart terminal/IDE after installing via package manager

**Issue: Port 9002 already in use**

- Change port: `npm run dev -- -p 3001`
- Or kill existing process: `lsof -ti:9002 | xargs kill -9`

**Issue: TensorFlow/CUDA errors**

- Use CPU version: `pip install tensorflow-cpu`
- For GPU: install CUDA 12.x and cuDNN 9.x, then `pip install tensorflow[and-cuda]`

### Development Commands

```bash
# Type checking
npm run typecheck

# Linting
npm run lint

# Genkit dev server (interactive flow testing)
npm run genkit:dev

# Genkit watcher (auto-reload on file changes)
npm run genkit:watch

# Build Next.js
npm run build
```

---

## ⚙️ Fusion & Decision Logic

Aegis combines agent outputs using a **weighted fusion strategy**:

- **Perceptual:** 65–85%
- **Semantic:** 15–35%
- Dynamic reweighting for provenance and adversarial results.

**Overrides:**

- Verified provenance → confidence ↑
- Spectral or temporal anomalies → confidence ↓
- Fragile robustness → confidence ↓

**Classification thresholds:**

- **Likely Real** (≥ 0.7)
- **Uncertain / Mixed Evidence** (0.4–0.7)
- **Likely AI-Generated** (< 0.4)

The **Reasoning Agent** then synthesizes a narrative and audit trail explaining the decision.

---

## 🧠 Deployment Blueprint

- **Flask CV Server:** Dockerized GPU container behind Nginx (TLS 1.2)
- **Orchestrator:** Firebase Functions + Vertex AI GenKit flows
- **Storage:** GCS/S3 with signed URLs & retention policy
- **Async Pipeline:** Job-based processing with polling/webhooks
- **Auth & Security:** API keys, PKI signatures, rate limits
- **Monitoring:** Prometheus, Grafana, Sentry
- **CI/CD:** GitHub Actions → Cloud Run auto-deploy

---

## 🎥 Dataset

The training and evaluation datasets include:

- Real videos captured from multiple camera models
- Synthetic samples generated using **Sora**, **Runway**, and **Pika**
- Adversarially perturbed variants for stress testing

**Dataset Link:** [Access Dataset Here](https://www.kaggle.com/datasets/kanzeus/realai-video-dataset)

---

## 🔐 Security, Ethics & Legal

- Probabilistic outputs: “Likely Real / Likely AI / Uncertain”
- Expert review for critical or ambiguous cases
- GDPR-compliant data handling and deletion flows
- Transparent labeling of limitations
- Abuse prevention through authentication and rate limits

---

## 🌍 Impact

Aegis empowers individuals, journalists, and organizations to **detect synthetic media** with explainability and confidence.  
By combining **signal forensics**, **semantic reasoning**, and **provenance analysis**, it provides a **trust framework for digital authenticity** in the generative era.

---

## 🎬 Demo Video

Watch the full **project explanation and live demo** on YouTube:  
**[▶ Watch Demo](https://youtu.be/1KvlXqhnYgE?si=ygZgFhgrXoiqt8Hr)**

---

**Team:** Ansh Jain, Kshaunish Harsha, Chaitanya Sawant, Joel Purohit  
**Built with:** TensorFlow, OpenCV, Flask, Firebase, Vertex AI GenKit  
**Domains:** AI Forensics · Deepfake Detection · Explainable AI · Trust & Safety
