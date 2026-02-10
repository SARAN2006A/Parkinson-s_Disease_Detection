import numpy as np
import pandas as pd
import os
import logging
from scipy.signal import welch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points a, b, c.
    b is the vertex.
    Points are [x, y] coordinates.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def get_euclidean_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def extract_kinematic_features(landmarks_seq, fps=30):
    """
    Extracts time-series features from a sequence of landmarks.
    landmarks_seq: dict of joint names to (N, 2) arrays.
    """
    features = {}
    
    # CPM Joint mapping (based on UDysRS dataset)
    # Ensure raw data matches this format: { 'joint_name': [[x,y], [x,y]...], ... }
    
    num_frames = len(landmarks_seq['neck']) if 'neck' in landmarks_seq else 0
    if num_frames < 10:
        return None

    # --- 1. Joint Angles ---
    # Elbow Angles (Lsho - Lelb - Lwri)
    if all(k in landmarks_seq for k in ['Lsho', 'Lelb', 'Lwri']):
        l_elbow_angles = [calculate_angle(landmarks_seq['Lsho'][i], landmarks_seq['Lelb'][i], landmarks_seq['Lwri'][i]) 
                          for i in range(num_frames)]
        features['L_Elbow_Angle_Mean'] = np.mean(l_elbow_angles)
        features['L_Elbow_Angle_Std'] = np.std(l_elbow_angles)
        
    if all(k in landmarks_seq for k in ['Rsho', 'Relb', 'Rwri']):
        r_elbow_angles = [calculate_angle(landmarks_seq['Rsho'][i], landmarks_seq['Relb'][i], landmarks_seq['Rwri'][i]) 
                          for i in range(num_frames)]
        features['R_Elbow_Angle_Mean'] = np.mean(r_elbow_angles)
        features['R_Elbow_Angle_Std'] = np.std(r_elbow_angles)

    # --- 2. Velocities (Speed) ---
    # Wrist Speed
    for side in ['L', 'R']:
        joint = f'{side}wri'
        if joint in landmarks_seq:
            coords = np.array(landmarks_seq[joint])
            # Calculate displacement between frames
            velocities = np.linalg.norm(np.diff(coords, axis=0), axis=1) * fps # pixels/sec
            features[f'{side}_Wrist_Speed_Mean'] = np.mean(velocities)
            features[f'{side}_Wrist_Speed_Max'] = np.max(velocities)
            features[f'{side}_Wrist_Speed_Std'] = np.std(velocities)
            
            # --- 3. Tremor (Frequency Analysis) ---
            # Analyze 3-7Hz band power in velocity signal
            if len(velocities) > fps: # Need enough data for PSD
                f, Pxx = welch(velocities, fs=fps, nperseg=min(len(velocities), fps*2))
                # Integrate power in 3-7Hz band
                tremor_power = np.sum(Pxx[(f >= 3) & (f <= 7)])
                total_power = np.sum(Pxx)
                rel_tremor_power = tremor_power / (total_power + 1e-6)
                features[f'{side}_Tremor_Power'] = rel_tremor_power

    # --- 4. Asymmetry ---
    if 'L_Wrist_Speed_Mean' in features and 'R_Wrist_Speed_Mean' in features:
        features['Wrist_Speed_Asymmetry'] = abs(features['L_Wrist_Speed_Mean'] - features['R_Wrist_Speed_Mean'])

    return features

def build_features(input_path, output_path):
    logging.info("Loading processed data...")
    df = pd.read_pickle(os.path.join(input_path, 'merged_cpm_data.pkl'))
    
    # Check if df is a list or DataFrame
    if isinstance(df, list):
        df = pd.DataFrame(df)
        
    logging.info(f"Loaded {len(df)} samples.")
    
    X_list = []
    y_updrs = []
    y_udysrs = []
    
    for idx, row in df.iterrows():
        # Preprocessing of landmarks structure
        # CPM Data usually loaded as { 'joint': [ [x,y,c]... ] } or similar
        # Need to ensure extraction logic matches the structure
        
        # 'landmarks' column is a dictionary: {'Lank': ..., 'Lelb': ...}
        # Each value is a list of coordinates
        landmarks = row['landmarks']
        
        # Extract features
        feats = extract_kinematic_features(landmarks)
        
        if feats:
            feats['Trial_ID'] = row['trial_id']
            feats['Task'] = row['task']
            
            # Scores
            scores = row['scores']
            updrs = scores.get('UPDRS_Total', np.nan)
            
            # Handle UDysRS - Summing subscores if list
            udysrs_raw = scores.get('UDysRS_Comm', scores.get('UDysRS_Drink', []))
            udysrs = np.sum(udysrs_raw) if isinstance(udysrs_raw, list) else np.nan
            
            X_list.append(feats)
            y_updrs.append(updrs)
            y_udysrs.append(udysrs)
            
    # Combine into DataFrame
    X_df = pd.DataFrame(X_list)
    X_df['UPDRS_Total'] = y_updrs
    X_df['UDysRS_Total'] = y_udysrs
    
    # Drop rows with NaN targets (if training for that target)
    logging.info(f"Extracted features for {len(X_df)} samples.")
    
    output_file = os.path.join(output_path, 'final_dataset.csv')
    X_df.to_csv(output_file, index=False)
    logging.info(f"Saved feature dataset to {output_file}")

if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    processed_path = os.path.join(project_dir, 'data', 'processed')
    
    build_features(processed_path, processed_path)
