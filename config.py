# Tennis Coach Application Configuration

# MediaPipe Pose Landmarks Mapping (exact names from teammate's code)
POSE_LANDMARKS = {
    0: 'NOSE',
    1: 'LEFT_EYE_INNER',
    2: 'LEFT_EYE',
    3: 'LEFT_EYE_OUTER',
    4: 'RIGHT_EYE_INNER',
    5: 'RIGHT_EYE',
    6: 'RIGHT_EYE_OUTER',
    7: 'LEFT_EAR',
    8: 'RIGHT_EAR',
    9: 'MOUTH_LEFT',
    10: 'MOUTH_RIGHT',
    11: 'LEFT_SHOULDER',
    12: 'RIGHT_SHOULDER',
    13: 'LEFT_ELBOW',
    14: 'RIGHT_ELBOW',
    15: 'LEFT_WRIST',
    16: 'RIGHT_WRIST',
    17: 'LEFT_PINKY',
    18: 'RIGHT_PINKY',
    19: 'LEFT_INDEX',
    20: 'RIGHT_INDEX',
    21: 'LEFT_THUMB',
    22: 'RIGHT_THUMB',
    23: 'LEFT_HIP',
    24: 'RIGHT_HIP',
    25: 'LEFT_KNEE',
    26: 'RIGHT_KNEE',
    27: 'LEFT_ANKLE',
    28: 'RIGHT_ANKLE',
    29: 'LEFT_HEEL',
    30: 'RIGHT_HEEL',
    31: 'LEFT_FOOT_INDEX',
    32: 'RIGHT_FOOT_INDEX'
}

# Application Settings
APP_CONFIG = {
    'host': '0.0.0.0',
    'port': 5001,
    'debug': True, 
    'profiles_dir': 'user_profiles',
    'video_fps': 30,
    'detection_confidence': 0.5,
    'tracking_confidence': 0.5
}

# Default Profile Template (using exact landmark names)
DEFAULT_PROFILE = {
    'username': 'Default User',
    'body_parts': {
        # Main pose landmarks (33 points)
        'NOSE': 1,
        'LEFT_EYE_INNER': 1,
        'LEFT_EYE': 1,
        'LEFT_EYE_OUTER': 1,
        'RIGHT_EYE_INNER': 1,
        'RIGHT_EYE': 1,
        'RIGHT_EYE_OUTER': 1,
        'LEFT_EAR': 1,
        'RIGHT_EAR': 1,
        'MOUTH_LEFT': 1,
        'MOUTH_RIGHT': 1,
        'LEFT_SHOULDER': 1,
        'RIGHT_SHOULDER': 1,
        'LEFT_ELBOW': 1,
        'RIGHT_ELBOW': 1,
        'LEFT_WRIST': 1,
        'RIGHT_WRIST': 1,
        'LEFT_PINKY': 1,
        'RIGHT_PINKY': 1,
        'LEFT_INDEX': 1,
        'RIGHT_INDEX': 1,
        'LEFT_THUMB': 1,
        'RIGHT_THUMB': 1,
        'LEFT_HIP': 1,
        'RIGHT_HIP': 1,
        'LEFT_KNEE': 1,
        'RIGHT_KNEE': 1,
        'LEFT_ANKLE': 1,
        'RIGHT_ANKLE': 1,
        'LEFT_HEEL': 1,
        'RIGHT_HEEL': 1,
        'LEFT_FOOT_INDEX': 1,
        'RIGHT_FOOT_INDEX': 1,
        # Right hand landmarks (21 points)
        'R_WRIST': 1,
        'R_THUMB_CMC': 1,
        'R_THUMB_MCP': 1,
        'R_THUMB_IP': 1,
        'R_THUMB_TIP': 1,
        'R_INDEX_FINGER_MCP': 1,
        'R_INDEX_FINGER_PIP': 1,
        'R_INDEX_FINGER_DIP': 1,
        'R_INDEX_FINGER_TIP': 1,
        'R_MIDDLE_FINGER_MCP': 1,
        'R_MIDDLE_FINGER_PIP': 1,
        'R_MIDDLE_FINGER_DIP': 1,
        'R_MIDDLE_FINGER_TIP': 1,
        'R_RING_FINGER_MCP': 1,
        'R_RING_FINGER_PIP': 1,
        'R_RING_FINGER_DIP': 1,
        'R_RING_FINGER_TIP': 1,
        'R_PINKY_MCP': 1,
        'R_PINKY_PIP': 1,
        'R_PINKY_DIP': 1,
        'R_PINKY_TIP': 1,
        # Left hand landmarks (21 points)
        'L_WRIST': 1,
        'L_THUMB_CMC': 1,
        'L_THUMB_MCP': 1,
        'L_THUMB_IP': 1,
        'L_THUMB_TIP': 1,
        'L_INDEX_FINGER_MCP': 1,
        'L_INDEX_FINGER_PIP': 1,
        'L_INDEX_FINGER_DIP': 1,
        'L_INDEX_FINGER_TIP': 1,
        'L_MIDDLE_FINGER_MCP': 1,
        'L_MIDDLE_FINGER_PIP': 1,
        'L_MIDDLE_FINGER_DIP': 1,
        'L_MIDDLE_FINGER_TIP': 1,
        'L_RING_FINGER_MCP': 1,
        'L_RING_FINGER_PIP': 1,
        'L_RING_FINGER_DIP': 1,
        'L_RING_FINGER_TIP': 1,
        'L_PINKY_MCP': 1,
        'L_PINKY_PIP': 1,
        'L_PINKY_DIP': 1,
        'L_PINKY_TIP': 1
    },
    'stroke_count': 0,
    'session_history': []
}

# Accessibility Presets with proper medical terminology
ACCESSIBILITY_PRESETS = {
    'left_below_elbow_amputee': {
        'name': 'Left Transradial Amputee',
        'description': 'Left below-elbow amputation - no left wrist/hand tracking',
        'enabled_parts': ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'RIGHT_WRIST', 'RIGHT_PINKY', 'RIGHT_INDEX', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
    },
    'right_below_elbow_amputee': {
        'name': 'Right Transradial Amputee',
        'description': 'Right below-elbow amputation - no right wrist/hand tracking',
        'enabled_parts': ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'LEFT_PINKY', 'LEFT_INDEX', 'LEFT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
    },
    'left_above_elbow_amputee': {
        'name': 'Left Transhumeral Amputee',
        'description': 'Left above-elbow amputation - no left arm tracking',
        'enabled_parts': ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST', 'RIGHT_PINKY', 'RIGHT_INDEX', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
    },
    'right_above_elbow_amputee': {
        'name': 'Right Transhumeral Amputee',
        'description': 'Right above-elbow amputation - no right arm tracking',
        'enabled_parts': ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST', 'LEFT_PINKY', 'LEFT_INDEX', 'LEFT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
    },
    'left_below_knee_amputee': {
        'name': 'Left Transtibial Amputee',
        'description': 'Left below-knee amputation - no left ankle/foot tracking',
        'enabled_parts': ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'RIGHT_ANKLE', 'RIGHT_HEEL', 'RIGHT_FOOT_INDEX']
    },
    'right_below_knee_amputee': {
        'name': 'Right Transtibial Amputee',
        'description': 'Right below-knee amputation - no right ankle/foot tracking',
        'enabled_parts': ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'LEFT_HEEL', 'LEFT_FOOT_INDEX']
    },
    'left_above_knee_amputee': {
        'name': 'Left Transfemoral Amputee',
        'description': 'Left above-knee amputation - no left leg tracking',
        'enabled_parts': ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE', 'RIGHT_HEEL', 'RIGHT_FOOT_INDEX']
    },
    'right_above_knee_amputee': {
        'name': 'Right Transfemoral Amputee',
        'description': 'Right above-knee amputation - no right leg tracking',
        'enabled_parts': ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_HEEL', 'LEFT_FOOT_INDEX']
    },
    'wheelchair_player': {
        'name': 'Wheelchair Player',
        'description': 'Upper body focus - no leg tracking',
        'enabled_parts': ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB']
    },
    'full_body': {
        'name': 'Full Body Tracking',
        'description': 'Complete pose analysis - all landmarks enabled',
        'enabled_parts': list(DEFAULT_PROFILE['body_parts'].keys())
    }
}
