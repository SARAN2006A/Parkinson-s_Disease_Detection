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
    These are mock heuristic bounds to give the UI dynamic feedback.
    """
    if not features or not isinstance(features, dict):
        return ["Unable to parse detailed kinematic features."]
        
    findings = []
    
    # Analyze arm swing asymmetry
    asymmetry = features.get('step_length_asymmetry', 0)
    if asymmetry > 0.15:
        findings.append(f"Left-right asymmetry observed in stride (Asym: {asymmetry:.2f})")
        
    # Tremor
    tremor_energy = features.get('tremor_energy_hands', 0)
    if tremor_energy > 0.8:
        findings.append("Resting tremor detected in upper extremities.")
        
    # Posture
    posture_angle = features.get('trunk_lean_angle', 0)
    if posture_angle > 15:
        findings.append(f"Stooped posture detected (Forward lean: {posture_angle:.1f}°)")
        
    # Speed
    cadence = features.get('cadence', 100) # steps per minute
    if cadence < 80:
        findings.append("Bradykinesia: Slowness of movement detected.")
        
    if not findings:
        findings.append("No significant motor abnormalities found in extracted features.")
        
    return findings
