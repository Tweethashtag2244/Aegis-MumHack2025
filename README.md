# 🛡️ Aegis — Autonomous AI Video Detection & Attribution

Aegis is an **agentic AI framework** that autonomously detects, attributes, and explains **AI-generated or manipulated videos**.  
By combining **signal-level forensics**, **semantic reasoning**, and **provenance verification**, Aegis delivers interpretable, evidence-backed authenticity assessments.

---

## 🚀 Overview

AI-generated videos from tools like **Sora**, **Runway**, **Pika**, and **Synthesia** are becoming increasingly lifelike, blurring the line between real and synthetic media.  
Aegis provides a transparent, explainable verification layer — helping individuals, journalists, and platforms **restore trust in digital content**.

Built as a **multi-agent system**, Aegis evaluates videos through specialized AI agents that analyze different modalities — from pixel noise and lighting to audio semantics and provenance.

---

## 🧩 Core Agents

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

**Dataset Link:** [Access Dataset Here](https://example.com/dataset-placeholder)

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
