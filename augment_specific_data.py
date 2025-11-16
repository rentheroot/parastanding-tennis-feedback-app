import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal, interpolate
from scipy.spatial import distance
import json

class TennisFeatureExtractor:
    """Extract ML-ready features from tennis forehand pose data."""
    
    def __init__(self, db_path):
        """Initialize extractor with path to SQLite database."""
        self.db_path = db_path
        self.data = self._load_and_normalize_data()
        self.num_frames = len(self.data) // 33  # 33 landmarks per frame
        
    def _load_and_normalize_data(self):
        """Load and normalize pose data from SQLite database."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM pose_data ORDER BY timestamp, landmark"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Normalize to fixed number of frames via interpolation if needed
        return df
    
    def _reshape_to_frames(self):
        """Reshape data into frames x landmarks structure."""
        landmarks = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
                    'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
                    'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
                    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
                    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
                    'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
                    'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
                    'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
                    'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
        
        frames = []
        for frame_idx in range(self.num_frames):
            frame_data = {}
            for landmark in landmarks:
                mask = self.data['landmark'] == landmark
                lm_data = self.data[mask].iloc[frame_idx] if mask.sum() > frame_idx else None
                if lm_data is not None:
                    frame_data[landmark] = {
                        'x': lm_data['x'],
                        'y': lm_data['y'],
                        'z': lm_data['z'],
                        'visibility': lm_data['visibility']
                    }
            frames.append(frame_data)
        return frames
    
    def _get_landmark_series(self, landmark_name, coord='x'):
        """Get time series for a specific landmark coordinate."""
        frames = self._reshape_to_frames()
        series = []
        for frame in frames:
            if landmark_name in frame:
                series.append(frame[landmark_name][coord])
            else:
                series.append(np.nan)
        return np.array(series)
    
    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points."""
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    def _calculate_velocity(self, series):
        """Calculate velocity from position series."""
        return np.diff(series)
    
    def _calculate_acceleration(self, series):
        """Calculate acceleration from position series."""
        velocity = self._calculate_velocity(series)
        return np.diff(velocity)
    
    # KINEMATIC FEATURES
    def extract_joint_angles(self):
        """Extract all relevant joint angles throughout swing."""
        frames = self._reshape_to_frames()
        features = {}
        
        # Right arm angles (hitting arm)
        elbow_angles = []
        shoulder_angles = []
        
        for frame in frames:
            if all(k in frame for k in ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']):
                elbow_angle = self._calculate_angle(
                    frame['RIGHT_SHOULDER'],
                    frame['RIGHT_ELBOW'],
                    frame['RIGHT_WRIST']
                )
                elbow_angles.append(elbow_angle)
            
            if all(k in frame for k in ['RIGHT_HIP', 'RIGHT_SHOULDER', 'RIGHT_ELBOW']):
                shoulder_angle = self._calculate_angle(
                    frame['RIGHT_HIP'],
                    frame['RIGHT_SHOULDER'],
                    frame['RIGHT_ELBOW']
                )
                shoulder_angles.append(shoulder_angle)
        
        # Knee angles
        right_knee_angles = []
        left_knee_angles = []
        
        for frame in frames:
            if all(k in frame for k in ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE']):
                knee_angle = self._calculate_angle(
                    frame['RIGHT_HIP'],
                    frame['RIGHT_KNEE'],
                    frame['RIGHT_ANKLE']
                )
                right_knee_angles.append(knee_angle)
            
            if all(k in frame for k in ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE']):
                knee_angle = self._calculate_angle(
                    frame['LEFT_HIP'],
                    frame['LEFT_KNEE'],
                    frame['LEFT_ANKLE']
                )
                left_knee_angles.append(knee_angle)
        
        # Statistical features from angles
        features['elbow_angle_min'] = np.min(elbow_angles) if elbow_angles else 0
        features['elbow_angle_max'] = np.max(elbow_angles) if elbow_angles else 0
        features['elbow_angle_mean'] = np.mean(elbow_angles) if elbow_angles else 0
        features['elbow_angle_std'] = np.std(elbow_angles) if elbow_angles else 0
        features['elbow_angle_range'] = features['elbow_angle_max'] - features['elbow_angle_min']
        
        features['shoulder_angle_min'] = np.min(shoulder_angles) if shoulder_angles else 0
        features['shoulder_angle_max'] = np.max(shoulder_angles) if shoulder_angles else 0
        features['shoulder_angle_mean'] = np.mean(shoulder_angles) if shoulder_angles else 0
        
        features['right_knee_angle_min'] = np.min(right_knee_angles) if right_knee_angles else 0
        features['right_knee_angle_max'] = np.max(right_knee_angles) if right_knee_angles else 0
        features['right_knee_flexion_range'] = features['right_knee_angle_max'] - features['right_knee_angle_min']
        
        features['left_knee_angle_min'] = np.min(left_knee_angles) if left_knee_angles else 0
        features['left_knee_angle_max'] = np.max(left_knee_angles) if left_knee_angles else 0
        
        return features
    
    def extract_velocity_features(self):
        """Extract velocity-based features (racket speed indicators)."""
        features = {}
        
        # Wrist velocity (proxy for racket head speed)
        wrist_x = self._get_landmark_series('RIGHT_WRIST', 'x')
        wrist_y = self._get_landmark_series('RIGHT_WRIST', 'y')
        
        wrist_velocity = np.sqrt(
            self._calculate_velocity(wrist_x)**2 + 
            self._calculate_velocity(wrist_y)**2
        )
        
        features['wrist_max_velocity'] = np.max(wrist_velocity)
        features['wrist_mean_velocity'] = np.mean(wrist_velocity)
        features['wrist_velocity_std'] = np.std(wrist_velocity)
        features['wrist_peak_velocity_frame'] = np.argmax(wrist_velocity) / len(wrist_velocity)  # Normalized
        
        # Elbow velocity
        elbow_x = self._get_landmark_series('RIGHT_ELBOW', 'x')
        elbow_y = self._get_landmark_series('RIGHT_ELBOW', 'y')
        
        elbow_velocity = np.sqrt(
            self._calculate_velocity(elbow_x)**2 + 
            self._calculate_velocity(elbow_y)**2
        )
        
        features['elbow_max_velocity'] = np.max(elbow_velocity)
        
        # Hip center velocity (body movement)
        left_hip_x = self._get_landmark_series('LEFT_HIP', 'x')
        right_hip_x = self._get_landmark_series('RIGHT_HIP', 'x')
        hip_center_x = (left_hip_x + right_hip_x) / 2
        
        hip_velocity = np.abs(self._calculate_velocity(hip_center_x))
        features['hip_max_velocity'] = np.max(hip_velocity)
        
        return features
    
    def extract_acceleration_features(self):
        """Extract acceleration features (impact and explosiveness)."""
        features = {}
        
        wrist_x = self._get_landmark_series('RIGHT_WRIST', 'x')
        wrist_y = self._get_landmark_series('RIGHT_WRIST', 'y')
        
        wrist_accel_x = self._calculate_acceleration(wrist_x)
        wrist_accel_y = self._calculate_acceleration(wrist_y)
        
        wrist_accel = np.sqrt(wrist_accel_x**2 + wrist_accel_y**2)
        
        features['wrist_max_acceleration'] = np.max(wrist_accel)
        features['wrist_max_deceleration'] = np.min(wrist_accel_x)  # Contact point indicator
        
        return features
    
    def extract_rotation_features(self):
        """Extract body rotation features (power generation)."""
        features = {}
        
        # Hip rotation
        left_hip_x = self._get_landmark_series('LEFT_HIP', 'x')
        right_hip_x = self._get_landmark_series('RIGHT_HIP', 'x')
        left_hip_y = self._get_landmark_series('LEFT_HIP', 'y')
        right_hip_y = self._get_landmark_series('RIGHT_HIP', 'y')
        
        hip_angles = np.arctan2(
            left_hip_y - right_hip_y,
            left_hip_x - right_hip_x
        )
        hip_angles_deg = np.degrees(hip_angles)
        
        features['hip_rotation_range'] = np.ptp(hip_angles_deg)  # Peak to peak
        features['hip_rotation_max'] = np.max(hip_angles_deg)
        features['hip_rotation_min'] = np.min(hip_angles_deg)
        features['hip_rotation_std'] = np.std(hip_angles_deg)
        
        # Shoulder rotation
        left_shoulder_x = self._get_landmark_series('LEFT_SHOULDER', 'x')
        right_shoulder_x = self._get_landmark_series('RIGHT_SHOULDER', 'x')
        left_shoulder_y = self._get_landmark_series('LEFT_SHOULDER', 'y')
        right_shoulder_y = self._get_landmark_series('RIGHT_SHOULDER', 'y')
        
        shoulder_angles = np.arctan2(
            left_shoulder_y - right_shoulder_y,
            left_shoulder_x - right_shoulder_x
        )
        shoulder_angles_deg = np.degrees(shoulder_angles)
        
        features['shoulder_rotation_range'] = np.ptp(shoulder_angles_deg)
        features['shoulder_rotation_max'] = np.max(shoulder_angles_deg)
        features['shoulder_rotation_min'] = np.min(shoulder_angles_deg)
        
        # Hip-shoulder separation (X-factor)
        separation = np.abs(hip_angles_deg - shoulder_angles_deg)
        features['max_hip_shoulder_separation'] = np.max(separation)
        features['mean_hip_shoulder_separation'] = np.mean(separation)
        
        return features
    
    def extract_weight_transfer_features(self):
        """Extract weight transfer and balance features."""
        features = {}
        
        # Forward/backward hip movement
        left_hip_x = self._get_landmark_series('LEFT_HIP', 'x')
        right_hip_x = self._get_landmark_series('RIGHT_HIP', 'x')
        hip_center_x = (left_hip_x + right_hip_x) / 2
        
        features['hip_forward_movement'] = hip_center_x[-1] - hip_center_x[0]
        features['hip_max_forward_position'] = np.max(hip_center_x)
        features['hip_min_forward_position'] = np.min(hip_center_x)
        features['hip_movement_range'] = features['hip_max_forward_position'] - features['hip_min_forward_position']
        
        # Vertical hip movement (staying low)
        left_hip_y = self._get_landmark_series('LEFT_HIP', 'y')
        right_hip_y = self._get_landmark_series('RIGHT_HIP', 'y')
        hip_center_y = (left_hip_y + right_hip_y) / 2
        
        features['hip_vertical_movement'] = np.ptp(hip_center_y)
        features['hip_lowest_point'] = np.min(hip_center_y)
        
        return features
    
    def extract_timing_features(self):
        """Extract timing and sequencing features (kinetic chain)."""
        features = {}
        
        # Find peak velocity timing for different body parts
        wrist_x = self._get_landmark_series('RIGHT_WRIST', 'x')
        wrist_y = self._get_landmark_series('RIGHT_WRIST', 'y')
        wrist_velocity = np.sqrt(
            self._calculate_velocity(wrist_x)**2 + 
            self._calculate_velocity(wrist_y)**2
        )
        
        elbow_x = self._get_landmark_series('RIGHT_ELBOW', 'x')
        elbow_y = self._get_landmark_series('RIGHT_ELBOW', 'y')
        elbow_velocity = np.sqrt(
            self._calculate_velocity(elbow_x)**2 + 
            self._calculate_velocity(elbow_y)**2
        )
        
        shoulder_x = self._get_landmark_series('RIGHT_SHOULDER', 'x')
        shoulder_y = self._get_landmark_series('RIGHT_SHOULDER', 'y')
        shoulder_velocity = np.sqrt(
            self._calculate_velocity(shoulder_x)**2 + 
            self._calculate_velocity(shoulder_y)**2
        )
        
        # Normalized timing (0-1)
        total_frames = len(wrist_velocity)
        features['wrist_peak_timing'] = np.argmax(wrist_velocity) / total_frames
        features['elbow_peak_timing'] = np.argmax(elbow_velocity) / total_frames
        features['shoulder_peak_timing'] = np.argmax(shoulder_velocity) / total_frames
        
        # Proper kinetic chain: shoulder -> elbow -> wrist
        features['elbow_wrist_timing_diff'] = features['wrist_peak_timing'] - features['elbow_peak_timing']
        features['shoulder_elbow_timing_diff'] = features['elbow_peak_timing'] - features['shoulder_peak_timing']
        
        return features
    
    def extract_pose_quality_features(self):
        """Extract features indicating pose tracking quality."""
        features = {}
        
        # Average visibility scores
        key_landmarks = ['RIGHT_WRIST', 'RIGHT_ELBOW', 'RIGHT_SHOULDER', 
                        'LEFT_HIP', 'RIGHT_HIP', 'RIGHT_KNEE', 'LEFT_KNEE']
        
        visibilities = []
        for landmark in key_landmarks:
            vis = self._get_landmark_series(landmark, 'visibility')
            visibilities.extend(vis[~np.isnan(vis)])
        
        features['mean_visibility'] = np.mean(visibilities) if visibilities else 0
        features['min_visibility'] = np.min(visibilities) if visibilities else 0
        
        return features
    
    def extract_spatial_features(self):
        """Extract spatial relationship features."""
        features = {}
        frames = self._reshape_to_frames()
        
        # Stance width
        stance_widths = []
        for frame in frames:
            if 'LEFT_ANKLE' in frame and 'RIGHT_ANKLE' in frame:
                width = abs(frame['LEFT_ANKLE']['x'] - frame['RIGHT_ANKLE']['x'])
                stance_widths.append(width)
        
        features['mean_stance_width'] = np.mean(stance_widths) if stance_widths else 0
        features['max_stance_width'] = np.max(stance_widths) if stance_widths else 0
        
        # Arm extension at peak
        wrist_x = self._get_landmark_series('RIGHT_WRIST', 'x')
        wrist_y = self._get_landmark_series('RIGHT_WRIST', 'y')
        shoulder_x = self._get_landmark_series('RIGHT_SHOULDER', 'x')
        shoulder_y = self._get_landmark_series('RIGHT_SHOULDER', 'y')
        
        arm_extensions = np.sqrt(
            (wrist_x - shoulder_x)**2 + 
            (wrist_y - shoulder_y)**2
        )
        
        features['max_arm_extension'] = np.max(arm_extensions)
        features['mean_arm_extension'] = np.mean(arm_extensions)
        
        return features
    
    def extract_all_features(self):
        """Extract all features and return as a single feature vector."""
        all_features = {}
        
        # Combine all feature sets
        all_features.update(self.extract_joint_angles())
        all_features.update(self.extract_velocity_features())
        all_features.update(self.extract_acceleration_features())
        all_features.update(self.extract_rotation_features())
        all_features.update(self.extract_weight_transfer_features())
        all_features.update(self.extract_timing_features())
        all_features.update(self.extract_pose_quality_features())
        all_features.update(self.extract_spatial_features())
        
        # Add metadata
        all_features['num_frames'] = self.num_frames
        all_features['clip_duration'] = self.num_frames / 30.0  # Assuming 30fps
        
        return all_features


def process_directory(directory_path, label):
    """
    Process all database files in a directory and extract features.
    
    Args:
        directory_path: Path to directory containing .db files
        label: Binary label (1 for successful, 0 for unsuccessful)
    
    Returns:
        DataFrame with features and labels
    """
    directory = Path(directory_path)
    db_files = list(directory.glob("*.db"))
    
    print(f"Processing {len(db_files)} files from {directory_path}...")
    
    all_features = []
    failed_files = []
    
    for db_file in db_files:
        try:
            extractor = TennisFeatureExtractor(str(db_file))
            features = extractor.extract_all_features()
            features['label'] = label
            features['filename'] = db_file.name
            all_features.append(features)
            print(f"  ✓ {db_file.name}")
        except Exception as e:
            print(f"  ✗ {db_file.name}: {e}")
            failed_files.append(db_file.name)
    
    if failed_files:
        print(f"\nFailed to process {len(failed_files)} files:")
        for f in failed_files:
            print(f"  - {f}")
    
    return pd.DataFrame(all_features)


def create_training_dataset(successful_dir, unsuccessful_dir, output_path="tennis_training_data.csv"):
    """
    Create a complete training dataset from successful and unsuccessful shot directories.
    
    Args:
        successful_dir: Directory containing successful shot databases
        unsuccessful_dir: Directory containing unsuccessful shot databases
        output_path: Path to save the combined dataset
    
    Returns:
        DataFrame with all features and labels
    """
    print("=" * 70)
    print("TENNIS FOREHAND FEATURE EXTRACTION FOR ML")
    print("=" * 70)
    
    # Process successful shots (label = 1)
    print("\n📊 Processing SUCCESSFUL shots...")
    successful_df = process_directory(successful_dir, label=1)
    print(f"Extracted features from {len(successful_df)} successful shots")
    
    # Process unsuccessful shots (label = 0)
    print("\n📊 Processing UNSUCCESSFUL shots...")
    unsuccessful_df = process_directory(unsuccessful_dir, label=0)
    print(f"Extracted features from {len(unsuccessful_df)} unsuccessful shots")
    
    # Combine datasets
    combined_df = pd.concat([successful_df, unsuccessful_df], ignore_index=True)
    
    # Shuffle the dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    combined_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Total samples: {len(combined_df)}")
    print(f"Successful shots: {len(successful_df)} ({len(successful_df)/len(combined_df)*100:.1f}%)")
    print(f"Unsuccessful shots: {len(unsuccessful_df)} ({len(unsuccessful_df)/len(combined_df)*100:.1f}%)")
    print(f"Total features: {len(combined_df.columns) - 2}")  # Exclude label and filename
    print(f"\nDataset saved to: {output_path}")
    
    # Display feature statistics
    print("\n" + "=" * 70)
    print("FEATURE STATISTICS")
    print("=" * 70)
    
    feature_cols = [col for col in combined_df.columns if col not in ['label', 'filename']]
    
    print("\nSample of extracted features:")
    print(combined_df[feature_cols[:10]].describe())
    
    print("\n" + "=" * 70)
    print("READY FOR MACHINE LEARNING!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Load the dataset with: df = pd.read_csv('tennis_training_data.csv')")
    print("2. Split into train/test sets")
    print("3. Normalize/standardize features")
    print("4. Train your neural network or other ML model")
    print("5. Evaluate performance and iterate!")
    
    return combined_df


def export_feature_metadata(df, output_path="feature_metadata.json"):
    """Export feature names and descriptions for documentation."""
    feature_cols = [col for col in df.columns if col not in ['label', 'filename']]
    
    metadata = {
        'total_features': len(feature_cols),
        'feature_groups': {
            'joint_angles': [f for f in feature_cols if 'angle' in f],
            'velocity': [f for f in feature_cols if 'velocity' in f],
            'acceleration': [f for f in feature_cols if 'acceleration' in f],
            'rotation': [f for f in feature_cols if 'rotation' in f],
            'weight_transfer': [f for f in feature_cols if 'hip' in f and 'rotation' not in f],
            'timing': [f for f in feature_cols if 'timing' in f],
            'spatial': [f for f in feature_cols if 'extension' in f or 'stance' in f],
            'quality': [f for f in feature_cols if 'visibility' in f]
        },
        'all_features': feature_cols
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Feature metadata exported to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Define your directories
    SUCCESSFUL_DIR = "clips/successful"
    UNSUCCESSFUL_DIR = "clips/unsuccessful"
    
    # Create training dataset
    dataset = create_training_dataset(
        successful_dir=SUCCESSFUL_DIR,
        unsuccessful_dir=UNSUCCESSFUL_DIR,
        output_path="tennis_training_data.csv"
    )
    
    # Export feature metadata
    export_feature_metadata(dataset, "feature_metadata.json")
    
    # Optional: Display correlation with label
    print("\n" + "=" * 70)
    print("TOP FEATURES CORRELATED WITH SUCCESS")
    print("=" * 70)
    feature_cols = [col for col in dataset.columns if col not in ['label', 'filename']]
    correlations = dataset[feature_cols].corrwith(dataset['label']).abs().sort_values(ascending=False)
    print("\nTop 15 features:")
    print(correlations.head(15))