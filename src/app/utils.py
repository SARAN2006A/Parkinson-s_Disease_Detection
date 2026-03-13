import numpy as np

def get_severity_level(score):
    """
    Returns the severity level (0-4) and a descriptive label based on the UPDRS score
    Score ranges from 0-132
    """
    try:
        score = float(score)
    except (ValueError, TypeError):
        return 0, "Normal"

    if score <= 10:
        return 0, "Normal/Slight"
    elif score <= 30:
        return 1, "Mild"
    elif score <= 50:
        return 2, "Moderate"
    elif score <= 80:
        return 3, "Severe"
    else:
        return 4, "Advanced"

def generate_key_findings(features):
    """
    Generates a list of string observations based on the extracted kinematic features.
    """
    if not features or not isinstance(features, dict):
        return ["Unable to parse detailed kinematic features."]
        
    findings = []
    
    # 1. Asymmetry
    asymmetry = features.get('Wrist_Speed_Asymmetry', 0)
    if asymmetry > 20: # velocity asymmetry in pixels/sec
        findings.append(f"Left-right asymmetry observed in arm swing (Asym: {asymmetry:.0f} px/s)")
        
    # 2. Tremor
    l_tremor = features.get('L_Tremor_Power', 0)
    r_tremor = features.get('R_Tremor_Power', 0)
    max_tremor = max(l_tremor, r_tremor)
    if max_tremor > 0.05: # Tremor power relative ratio
        findings.append(f"Resting tremor detected in upper extremities (Power ratio: {max_tremor:.2f})")
        
    # 3. Speed (Bradykinesia)
    l_speed = features.get('L_Wrist_Speed_Mean', 2000)
    r_speed = features.get('R_Wrist_Speed_Mean', 2000)
    avg_speed = (l_speed + r_speed) / 2
    if avg_speed < 1100: # pixels/sec
        findings.append(f"Bradykinesia: Slowness of movement detected.")
        
    if not findings:
        findings.append("No significant motor abnormalities found in extracted features.")
        
    return findings
