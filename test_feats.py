import numpy as np
from src.features.build_features import extract_kinematic_features

# Simulate the demo mode in inference.py
def simulate(is_pd_video, task_name="Gait Analysis"):
    landmarks_seq = {k: [] for k in ['Lsho', 'Rsho', 'Lelb', 'Relb', 'Lwri', 'Rwri', 
                                    'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank', 
                                    'head', 'face', 'neck']}
    fps = 30.0
    speed_factor = 1.2
    tremor_amp = 0.5
    stiffness = 1.0
    
    if is_pd_video:
        speed_factor = 0.5  
        tremor_amp = 8.0    
        stiffness = 0.6     
    else:
        tremor_amp = 0.1    

    if task_name == "Tremor Analysis":
        tremor_amp = 20.0 if is_pd_video else 5.0
        stiffness = 1.0
    elif task_name == "Hand Movements":
        tremor_amp = 15.0 if is_pd_video else 2.0
        speed_factor = 0.4 if is_pd_video else 1.2

    num_frames = 100
    t = np.linspace(0, 10 * speed_factor, num_frames)
    
    for i in range(num_frames):
        tremor_L = np.random.normal(0, tremor_amp)
        tremor_R = np.random.normal(0, tremor_amp)
        
        landmarks_seq['Lwri'].append([300 + (50 * stiffness * np.sin(t[i])) + tremor_L, 500])
        landmarks_seq['Rwri'].append([350 - (50 * stiffness * np.sin(t[i])) + tremor_R, 500])
        landmarks_seq['Lsho'].append([300, 200])
        landmarks_seq['Rsho'].append([400, 200])
        landmarks_seq['Lelb'].append([300, 350])
        landmarks_seq['Relb'].append([400, 350])
        landmarks_seq['neck'].append([350, 200])
        landmarks_seq['head'].append([350, 150]) 
        landmarks_seq['face'].append([350, 150])
        
        landmarks_seq['Lhip'].append([320, 500])
        landmarks_seq['Rhip'].append([380, 500])
        landmarks_seq['Lkne'].append([320, 700])
        landmarks_seq['Rkne'].append([380, 700])
        landmarks_seq['Lank'].append([320, 900])
        landmarks_seq['Rank'].append([380, 900])

    valid_seq = {k: np.array(v) for k, v in landmarks_seq.items() if len(v) > 10}
    
    features = extract_kinematic_features(valid_seq, fps=fps)
    return features

print("Healthy:", simulate(False))
print("PD Gait:", simulate(True, "Gait Analysis"))
print("PD Tremor:", simulate(True, "Tremor Analysis"))
