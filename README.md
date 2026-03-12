# NeuroGait AI: Video-based Parkinson's Disease Estimation

A production-ready artificial intelligence platform designed to estimate the severity of Parkinson’s Disease and Levodopa-induced Dyskinesia from single-person videos. This project leverages Pose Estimation (MediaPipe) and a Random Forest classification pipeline, completely eliminating the need for expensive wearable hardware sensors.

## ✨ Key Features

- **Non-invasive Assessment**: Evaluates motor symptoms using standard 2D video, eliminating the need for wearable sensors.
- **Advanced Pose Tracking**: Uses Google's MediaPipe for fast, accurate skeletal landmark extraction.
- **Machine Learning Pipeline**: Uses a trained Random Forest classifier to predict UPDRS severity scores.
- **Modern Web Interface**: A sleek, responsive dashboard featuring a modern dark theme and dynamic status updates.
- **Cloud-Ready Back-End**: Fast and scalable REST API powered by Flask, ready for local execution or PaaS deployment.

---

## 🚀 System Architecture and Flow

The application features a decoupled, modern architecture optimized for both local inference and scalable cloud deployments.

### 1. Frontend Client (Browser)
- Built with standard HTML5, dynamic JavaScript (`app.js`), and a custom "glassmorphism" CSS framework (`style.css`).
- Provides an intuitive drag-and-drop interface for clinical staff to upload patient assessment videos safely.
- Renders responsive Gauge Charts (via Chart.js) bounding the estimated UPDRS scales dynamically.

### 2. Backend API (Flask Server)
- **Routing Engine:** A single scalable Python Flask server (`src/api/app.py`) serves the static frontend assets and handles the heavy data pipelines via a dedicated `/api/analyze` REST endpoint.
- **Pose Extraction:** Once a video is received, the backend utilizes **MediaPipe Pose** to chronologically track and record spatial body landmarks across the video's frames.
- **Kinematic Feature Engineering:** The application processes the raw landmarks to compute vital physiological metrics, including temporal speed, joint angle deviations, asymmetry, and tremor oscillations.
- **Random Forest Inference:** The extracted feature vectors are passed into a pre-trained scikit-learn Random Forest model (`VideoPredictor`), outputting a severity score aligned with the UPDRS (Unified Parkinson's Disease Rating Scale).

---

## 📋 Prerequisites

Before setting up the project locally, please ensure you have the following installed:
- **Python 3.9+** (Python 3.10 recommended)
- **Git** for version control
- A modern web browser to view the client interface

---

## 🏃 Local Development and Testing

If you are a developer looking to run or improve the model locally:

1. **Clone and Setup:**
   ```bash
   git clone <repository_url>
   cd parkinson
   pip install -r requirements.txt
   ```

2. **Start the Integrated Server:**
   The Flask server natively serves the web UI alongside the API endpoints.
   ```bash
   python src/api/app.py
   ```

3. **Access the Application:**
   Open your preferred web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```
   Upload a sample `.mp4` video (e.g., from `data/`) to test the inference pipeline.

---

## ☁️ Production Deployment (Render)

This repository is configured for 1-click PaaS deployments (such as Render.com) via the included `render.yaml` specification using `gunicorn`.

1. Connect your GitHub repository to Render.
2. Render will automatically detect the `render.yaml` Blueprint.
3. The platform will provision a Python 3.10 environment, install dependencies, and launch the WSGI `gunicorn` web server targeting `src.api.app:app`. 

## 📁 Repository Structure

```
parkinson/
├── data/               # Hidden directory containing secure patient raw data
├── render.yaml         # Render blueprint configuration for CI/CD
├── requirements.txt    # Python dependencies (Flask, Gunicorn, MediaPipe, scikit-learn)
├── src/                # Project Source Code
│   ├── api/            # Flask Web Server
│   │   ├── static/     # Compressed CSS and JS for the frontend
│   │   ├── templates/  # Core HTML views
│   │   └── app.py      # Main backend routing application
│   ├── data/           # ETL scripts for pre-processing the raw data
│   ├── features/       # Feature extraction algorithms (Kinematics)
│   ├── models/         # Inference definitions (`inference.py`)
│   └── visualization/  # Plotting algorithms
└── README.md           # Master documentation file
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are always welcome! Feel free to check the issues page and submit a Pull Request if you would like to help improve the system.

## 📄 License & Disclaimer

This project is developed for research and educational purposes. Always ensure compliance with local healthcare and data privacy regulations (such as HIPAA) when processing and storing patient video data.
