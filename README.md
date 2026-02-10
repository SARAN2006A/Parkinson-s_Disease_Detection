# Video-based Parkinson’s Disease and Dyskinesia Severity Estimation

A machine learning project to estimate the severity of Parkinson’s Disease and Levodopa-induced Dyskinesia from single-person videos using Pose Estimation (MediaPipe) and Random Forest, without wearable sensors.

## 🚀 Project Workflow (How it will work)

This system is designed to take a video as input and provide a clinical assessment as output.

### 1. Input (What the User Provides)
*   **A Video File (.mp4, .avi):** A recording of a person performing a specific motor task (e.g., walking, finger tapping, drinking from a cup).
*   **Target Task:** The user selects which task is being performed (e.g., "Leg Agility").

### 2. Processing (What the System Does)
1.  **Pose Extraction:** The system uses **MediaPipe Pose** to detect body landmarks (joints) in every frame of the video.
2.  **Feature Calculation:** It calculates kinematic features (speed, angles, tremors) from these landmarks.
3.  **Prediction:** These features are fed into a trained **Random Forest Model**.

### 3. Output (What the User Gets)
*   **Diagnosis Classification:** "Parkinson's Detected" vs. "Healthy Control".
*   **Severity Score:** A numerical score estimating the disease severity (e.g., **UPDRS Score: 12/100**).
*   **Visual Feedback:** The input video with pose landmarks overlaid, and graphs showing movement analysis.

---

## 🏃 How to Run

1.  **Activate Environment:** Ensure you have Python installed and dependencies set up.
2.  **Run Prediction:**
    ```bash
    python run_pipeline.py path/to/your/video.mp4
    ```
3.  **Example:**
    ```bash
    python run_pipeline.py test_video.mp4
    ```
    *(Note: If MediaPipe is not configured, this will run in Demo Mode and generate a simulated report).*

---

## 🌐 Launch Web App

You can also use the graphical interface:
```bash
streamlit run src/app/main.py
```
This will open the application in your browser.

## Project Structure

```
parkinson/
├── data/               # Data directory (ignored in git)
├── notebooks/          # Exploratory Jupyter notebooks
├── src/                # Source code
│   ├── data/           # Data processing scripts
│   ├── features/       # Feature extraction scripts
│   ├── models/         # Model training scripts
│   └── visualization/  # Visualization scripts
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Setup

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Place raw data in `data/raw/`.
