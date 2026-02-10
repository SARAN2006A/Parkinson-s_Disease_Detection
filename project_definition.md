# Project Definition: Video-based Parkinson’s Disease and Dyskinesia Severity Estimation

**Project Title:** Video-based Parkinson’s Disease and Dyskinesia Severity Estimation using Pose Estimation and Machine Learning

**Dataset:** UDysRS–UPDRS Parkinson’s Dataset (University of Toronto – Babak Taati)

## 1. Problem Statement

Parkinson's Disease (PD) and Levodopa-induced Dyskinesia (LID) necessitate frequent and objective monitoring to optimize medication dosage and manage symptoms effectively. Traditional clinical assessments, such as the Unified Parkinson's Disease Rating Scale (UPDRS) and the Unified Dyskinesia Rating Scale (UDysRS), are subjective, intermittent, and require specialized clinicians, leading to limited accessibility and potential variability in scoring. Current automated solutions often rely on intrusive wearable sensors, which can be cumbersome for patients, or require computationally intensive deep learning models that lack interpretability and demand significant hardware resources. There is a need for a non-invasive, efficient, and interpretable system that can estimate disease severity from standard video recordings using accessible computer vision and machine learning techniques.

## 2. Objectives

The primary objective of this project is to develop a robust, video-based system for estimating the severity of Parkinson’s Disease and Dyskinesia without the use of wearable sensors or deep end-to-end video classification networks.

**Specific Objectives:**

1.  **Develop a Video Processing Pipeline:** Implement a pipeline to ingest single-person video recordings and extract human pose landmarks using MediaPipe Pose.
2.  **Extract Interpretable Kinematic Features:** Design and compute clinically relevant features from the extracted pose landmarks, such as joint angles, angular velocities, tremor frequency, and movement amplitude, which correlate with PD and LID symptoms.
3.  **Build a Machine Learning Model:** Train and validate a Random Forest model (Regressor/Classifier) to map the extracted kinematic features to the corresponding clinical severity scores (UPDRS/UDysRS) provided in the dataset.
4.  **Evaluate System Performance:** Assess the accuracy and reliability of the severity estimation using standard regression metrics (e.g., Mean Absolute Error, Root Mean Squared Error) and correlation analysis against ground truth clinical ratings.
5.  **Ensure Computational Efficiency:** Optimize the feature extraction and modeling process to run effectively on standard consumer hardware (CPU/GPU) without relying on heavy deep learning architectures for video classification.

## 3. Scope

### Included
*   **Input Data Processing:**
    *   Loading and preprocessing of video files from the UDysRS–UPDRS dataset.
    *   Frame-by-frame extraction of 33 distinct body landmarks using MediaPipe Pose.
*   **Feature Engineering:**
    *   Calculation of spatiotemporal features: joint angles (elbow, knee, shoulder), velocity, acceleration, and jerk.
    *   Frequency domain analysis (FFT/PSD) to detect tremors (typical frequency range 3-6 Hz).
    *   Statistical aggregation of features (mean, std, max, skewness) over video segments.
*   **Machine Learning:**
    *   Usage of Random Forest Regressor and Classifier (Scikit-Learn).
    *   Feature selection and importance analysis to identify key predictors of severity.
    *   Hyperparameter tuning (Grid Search / Random Search) for model optimization.
*   **Evaluation:**
    *   Cross-validation (e.g., Leave-One-Subject-Out) to ensure generalizability.
    *   Comparison of predicted scores vs. clinician-provided labels.

### Excluded
*   **Hardware / Sensors:**
    *   **No Wearables:** The project will strictly exclude any accelerometer, gyroscope, or EMG data. Only video data will be used.
*   **Deep Learning Architectures:**
    *   **No End-to-End Video Classification:** Direct 3D-CNNs (e.g., I3D, C3D) or Video Transformers on raw pixels are excluded. The focus is on *pose-based* feature extraction followed by classical ML (Random Forest).
    *   Rationale: This ensures interpretability and lower computational cost.
*   **Deployment:**
    *   Real-time mobile app deployment is out of scope for the initial research phase.
    *   Medical diagnosis (system provides *estimation* of severity, not a confirmed medical diagnosis).

## 4. Methodology Overview

1.  **Video Input** -> **MediaPipe Pose** -> **Landmark Coordinates (x, y, z, visibility)**
2.  **Landmarks** -> **Feature Extraction** (Angles, Velocities, Tremor frequency) -> **Feature Vector**
3.  **Feature Vector** -> **Random Forest Model** -> **Severity Score (UPDRS/UDysRS)**
