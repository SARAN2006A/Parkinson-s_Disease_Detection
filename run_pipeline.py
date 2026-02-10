import argparse
import sys
import os
import logging
from src.models.inference import VideoPredictor

# Configure logging to show only INFO and above
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    parser = argparse.ArgumentParser(description="Parkinson's Severity Estimation from Video")
    parser.add_argument('video_path', type=str, help='Path to the input video file (mp4, avi)')
    args = parser.parse_args()
    
    video_path = args.video_path
    
    if not os.path.exists(video_path):
        print(f"Error: File not found at {video_path}")
        return

    print("="*60)
    print("Video-based Parkinson's Severity Estimation")
    print("="*60)
    print(f"Processing video: {video_path}")
    
    # Initialize Predictor
    project_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(project_dir, 'models')
    
    try:
        predictor = VideoPredictor(model_dir)
        
        # Run Prediction
        print("Running pose estimation and analysis...")
        prediction, features = predictor.predict_video(video_path)
        
        if prediction is None:
            print("Error: Could not process video.")
            return
            
        updrs_score = prediction
        
        # Interpret Score
        # UPDRS Part III (Motor) range: 0-132. 
        # Typically < 10 is considered normal/mild in many contexts suitable for screening, 
        # but clinically 0 is normal. 
        # Since our model trains on patients, it likely predicts within patient range (10-80).
        # We need a threshold. Let's use 15 as a safe demo threshold.
        
        diagnosis = "Parkinson's Features Detected" if updrs_score > 15 else "Normal / Unlikely Parkinson's"
        confidence = "High" # Placeholder, RF doesn't give confidence easily for regression without variance
        
        print("\n" + "="*60)
        print("ANALYSIS REPORT")
        print("="*60)
        print(f"Diagnosis: {diagnosis}")
        print(f"Estimated UPDRS Motor Score: {updrs_score:.2f} / 132")
        print("-" * 60)
        print("Key Features Analyzed:")
        if features:
            print(f" - Average Movement Speed: {features.get('R_Wrist_Speed_Mean', 0):.2f} px/sec")
            print(f" - Tremor Activity (3-7Hz): {features.get('R_Tremor_Power', 0):.4f}")
        print("="*60)
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
