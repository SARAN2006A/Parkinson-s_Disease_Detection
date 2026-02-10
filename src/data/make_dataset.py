import os
import json
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json_file(filepath):
    """Loads a JSON file (or JSON-formatted text file)."""
    with open(filepath, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {filepath}: {e}")
            return None

def parse_trial_id(key):
    """
    Parses trial ID from the trajectory key.
    Format is typically 'TrialID-SegmentID'.
    Example: '26-1' -> TrialID '26'
    """
    if '-' in key:
        return key.split('-')[0]
    return key

def process_data(raw_data_path, output_path):
    """
    Loads raw CPM trajectories and clinical ratings, merges them, and saves to processed folder.
    """
    
    # Paths
    comm_file = os.path.join(raw_data_path, 'Communication_all_export.txt')
    drink_file = os.path.join(raw_data_path, 'Drinking_all_export.txt')
    # leg_file = os.path.join(raw_data_path, 'LA_split_all_export.txt') # Different format, handling later
    
    udysrs_file = os.path.join(raw_data_path, 'UDysRS.txt')
    updrs_file = os.path.join(raw_data_path, 'UPDRS.txt')

    # Load Data
    logging.info("Loading Raw Data...")
    comm_data = load_json_file(comm_file)
    drink_data = load_json_file(drink_file)
    udysrs_data = load_json_file(udysrs_file)
    updrs_data = load_json_file(updrs_file)

    if not (comm_data and drink_data and udysrs_data and updrs_data):
        logging.error("Failed to load one or more data files.")
        return

    # Helper to merge data
    # We want a list of: { 'trial_id': ..., 'task': ..., 'landmarks': ..., 'scores': ... }
    
    processed_samples = []

    # Process Communication Task
    logging.info("Processing Communication Task...")
    for key, val in comm_data.items():
        trial_id = parse_trial_id(key)
        
        # Get Scores (if available) - Focusing on UPDRS Total and UDysRS Total for now
        # Note: Ratings dictionaries use the Trial ID as key
        
        scores = {}
        
        # UPDRS Part III Total (Motor Examination)
        if trial_id in updrs_data.get('Total', {}):
             scores['UPDRS_Total'] = updrs_data['Total'][trial_id] # Fixed: Is a float/int, not list
        
        # UDysRS Total
        # UDysRS structure needs inspection, assuming similar to UPDRS or has sub-sections
        # Based on file inspection, UDysRS.txt has keys like 'Communication', 'Drinking'
        if 'Communication' in udysrs_data and trial_id in udysrs_data['Communication']:
             # UDysRS often has multiple subscores. Let's store them or sum them?
             # For now, let's store the raw list
             scores['UDysRS_Comm'] = udysrs_data['Communication'][trial_id]

        # Only add if we have at least one score
        if scores:
            sample = {
                'trial_id': trial_id,
                'task': 'Communication',
                'segment': key,
                'landmarks': val['position'], # CPM Landmarks (15 joints)
                'scores': scores
            }
            processed_samples.append(sample)

    # Process Drinking Task
    logging.info("Processing Drinking Task...")
    for key, val in drink_data.items():
        trial_id = parse_trial_id(key)
        scores = {}

        if trial_id in updrs_data.get('Total', {}):
             scores['UPDRS_Total'] = updrs_data['Total'][trial_id]

        if 'Drinking' in udysrs_data and trial_id in udysrs_data['Drinking']:
             scores['UDysRS_Drink'] = udysrs_data['Drinking'][trial_id]

        if scores:
            sample = {
                'trial_id': trial_id,
                'task': 'Drinking',
                'segment': key,
                'landmarks': val['position'],
                'scores': scores
            }
            processed_samples.append(sample)

    logging.info(f"Total samples processed: {len(processed_samples)}")
    
    # Save as Pickle (preserves nested dictionary structure for landmarks)
    # Allows easier loading for Feature Engineering than CSV
    output_file = os.path.join(output_path, 'merged_cpm_data.pkl')
    pd.to_pickle(processed_samples, output_file)
    logging.info(f"Saved merged dataset to {output_file}")


if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    raw_path = os.path.join(project_dir, 'data', 'raw', 'UDysRS_UPDRS_Export')
    processed_path = os.path.join(project_dir, 'data', 'processed')
    
    process_data(raw_path, processed_path)
