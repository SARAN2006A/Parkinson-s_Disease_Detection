
import plotly.graph_objects as go

def get_severity_level(score):
    """
    Maps UPDRS score (0-132) to a 0-4 severity scale.
    
    Level 0: Normal -> Score 0-10
    Level 1: Slight -> Score 11-20
    Level 2: Mild -> Score 21-30
    Level 3: Moderate -> Score 31-50
    Level 4: Severe -> Score > 50
    """
    if score <= 10:
        return 0, "Normal"
    elif score <= 20:
        return 1, "Slight"
    elif score <= 30:
        return 2, "Mild"
    elif score <= 50:
        return 3, "Moderate"
    else:
        return 4, "Severe"

def get_severity_color(level):
    """
    Returns a hex color code based on severity level.
    """
    colors = {
        0: "#28a745",  # Green
        1: "#ffc107",  # Yellow
        2: "#fd7e14",  # Orange
        3: "#dc3545",  # Red
        4: "#850000"   # Dark Red
    }
    return colors.get(level, "#6c757d") # Default Grey

def generate_key_findings(features):
    """
    Generates human-readable key findings based on feature values.
    Assumes features is a dictionary or Series with kinetic data.
    """
    findings = []
    
    # 1. Speed Analysis (Bradykinesia)
    # Average L and R wrist speed if available
    l_speed = features.get('L_Wrist_Speed_Mean', 0)
    r_speed = features.get('R_Wrist_Speed_Mean', 0)
    avg_speed = (l_speed + r_speed) / 2 if (l_speed or r_speed) else 0
    
    # Thresholds tuned for 30fps video processing
    # Normal > 80 px/sec
    if avg_speed > 0 and avg_speed < 40.0: 
        findings.append(f"Significantly reduced movement speed (Bradykinesia) ({avg_speed:.1f} px/s)")
    elif avg_speed > 0 and avg_speed < 80.0:
        findings.append(f"Low tapping/movement speed ({avg_speed:.1f} px/s)")
        
    # 2. Rhythm/Asymmetry
    asymmetry = features.get('Wrist_Speed_Asymmetry', 0)
    if asymmetry > 50.0: 
        findings.append(f"Significant movement asymmetry detected")
        
    # 3. Tremor (Frequency Analysis)
    l_tremor = features.get('L_Tremor_Power', 0)
    r_tremor = features.get('R_Tremor_Power', 0)
    max_tremor = max(l_tremor, r_tremor)
    
    # White noise has flat PSD, so relative power in 3-7Hz band (4Hz width / 15Hz Nyquist) is ~0.26
    # Real tremor concentrates power, so we expect > 0.3
    if max_tremor > 0.3: 
        findings.append(f"Tremor detected in movement (Power: {max_tremor:.2f})")

    # 4. Range of Motion (Hypokinesia Check using Elbow Angle Std)
    l_rom = features.get('L_Elbow_Angle_Std', 0)
    r_rom = features.get('R_Elbow_Angle_Std', 0)
    avg_rom = (l_rom + r_rom) / 2
    
    # 5.0 degrees is a conservative threshold for "active" movement tests
    if avg_rom > 0 and avg_rom < 5.0:
        findings.append(f"Reduced range of motion (Hypokinesia) (ROM: {avg_rom:.1f}°)")

    # Fallback if no specific findings
    if not findings:
        findings.append("No significant movement anomalies detected.")
        
    return findings

def create_gauge_chart(score, max_score=132):
    """
    Creates a gauge chart using Plotly.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Severity Level"},
        gauge = {
            'axis': {'range': [None, 50], 'tickwidth': 1, 'tickcolor': "darkblue"}, # Cap at 50 for visual clarity of "Severity", even if UPDRS goes higher
            'bar': {'color': "black"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 10], 'color': '#28a745'},
                {'range': [10, 20], 'color': '#ffc107'},
                {'range': [20, 30], 'color': '#fd7e14'},
                {'range': [30, 50], 'color': '#dc3545'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score}}))
    
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
    return fig
